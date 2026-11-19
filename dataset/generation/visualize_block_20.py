"""Build a reference-fitted PCA48 viewer for five SD3.5 Block-20 trajectories.

Overview
--------
This script visualizes early denoising representations (0%, 10%, 20%, 30%, 40%, 50%)
from transformer block 20 of Stable Diffusion 3.5 Medium for five consecutive
replicates belonging to a single experimental cell ``(object_class, target_count)``.

To avoid overfitting the visualization basis to the images being inspected, PCA is
fitted exclusively on *independent reference cells* (cells ``0 .. reference_cell_stop - 1``
excluding ``--cell``), using randomly subsampled pixels from their earliest three
stages (0%, 10%, 20%). All color scales (RGB bounds and cumulative energy ceilings)
are likewise calibrated solely on those reference scores.

Outputs
-------
Given ``--html <path/to/viewer.html>``, the script writes three sibling files:
    1. ``<stem>.html``:
       Self-contained HTML viewer populated from ``block20_viewer_template.html`` with
       inline base64 PNGs (5 final thumbnails + 30 stages x 33 feature visualizations).
    2. ``<stem>.pca48.safetensors``:
       Reusable PCA48 projection basis (``mean``, ``components``) and reference-derived
       color normalization quantiles.
    3. ``<stem>.pca48.json``:
       Provenance metadata, reference image IDs, cumulative explained variance curve,
       and exact scale constants.
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from generate_v3 import EARLY_PERCENTAGES, MODEL_ID, SHARDS, read_manifest


# Load the sibling HTML template once at import time. It is expected to contain a
# literal `__DATA__` placeholder (typically inside a <script> tag) that receives
# the serialized viewer payload JSON.
HTML = Path(__file__).with_name("block20_viewer_template.html").read_text(encoding="utf-8")


def png_uri(image: Image.Image) -> str:
    """Encode a PIL Image as an inline base64 PNG data URI for embedding in HTML.

    Parameters
    ----------
    image : PIL.Image.Image
        RGB image (either a 48x48 PCA visualization or a <=512x512 final thumbnail).

    Returns
    -------
    str
        ``"data:image/png;base64,..."`` string suitable for ``<img src="...">``.
    """
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def rgb_map(scores: np.ndarray, low: np.ndarray, high: np.ndarray) -> Image.Image:
    """Map 3-channel PCA scores (PC1, PC2, PC3) to an 8-bit RGB image.

    Each channel is affinely rescaled from ``[low[c], high[c]]`` -> ``[0, 1]``,
    clipped to handle out-of-range outliers, and quantized to ``uint8`` in ``[0, 255]``.

    Parameters
    ----------
    scores : np.ndarray
        Spatial score slice of shape ``(H, W, 3)`` (here ``(48, 48, 3)``).
    low : np.ndarray
        Per-channel lower bound (1st reference percentile) of shape ``(3,)``.
    high : np.ndarray
        Per-channel upper bound (99th reference percentile) of shape ``(3,)``.

    Returns
    -------
    PIL.Image.Image
        ``(48, 48)`` RGB PIL image.
    """
    # Floor the dynamic range at 1e-6 to prevent division by zero on constant axes.
    span = np.maximum(high - low, 1e-6)
    colors = np.clip((scores - low) / span, 0, 1)
    return Image.fromarray(np.rint(colors * 255).astype(np.uint8), "RGB")


def energy_map(values: np.ndarray, scale: float) -> Image.Image:
    """Render nonnegative PCA score norm with a fixed, monotone color ramp.

    Normalizes a 2D scalar field by ``scale`` into ``[0, 1]`` and piecewise-linearly
    interpolates across 5 perceptually ordered RGB stops (deep indigo -> violet ->
    teal -> emerald -> bright yellow, similar to Viridis).

    Parameters
    ----------
    values : np.ndarray
        2D array of nonnegative L2 norms of shape ``(H, W)`` (here ``(48, 48)``).
    scale : float
        Upper reference quantile (99th percentile) mapped to 1.0 (brightest yellow).

    Returns
    -------
    PIL.Image.Image
        ``(48, 48)`` RGB PIL image.
    """
    # Normalize scalar energies into t in [0.0, 1.0], shape (H, W).
    t = np.clip(values / max(scale, 1e-6), 0, 1)

    # 5 RGB anchor stops of shape (5, 3) defining 4 uniform segments over [0, 1].
    anchors = np.array([
        [13, 8, 30], [48, 26, 101], [32, 110, 151],
        [40, 177, 139], [244, 230, 87],
    ], dtype=np.float32)

    # Map t in [0, 1] to continuous segment coordinates in [0, 4].
    position = t * (len(anchors) - 1)
    # Clamp left anchor index to {0, 1, 2, 3} so `left + 1` stays within bounds at t == 1.0.
    left = np.minimum(position.astype(np.int32), len(anchors) - 2)
    # Fractional interpolation weight within the active segment: shape (H, W, 1).
    fraction = (position - left)[..., None]
    # Linear blend between adjacent anchors: anchors[left] has shape (H, W, 3).
    colors = anchors[left] * (1 - fraction) + anchors[left + 1] * fraction
    return Image.fromarray(np.rint(colors).astype(np.uint8), "RGB")


def main() -> None:
    """Load target & reference features, fit reference PCA48, and write HTML + basis."""
    # -----------------------------------------------------------------------
    # 1. Parse & Validate Command-Line Arguments
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("prompts/numeracy_smoke_v3.csv"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--html", type=Path, default=Path("block20-cell02-cumulative.html"))
    parser.add_argument("--cell", type=int, default=2, help="Object/count cell, 0-39")
    parser.add_argument("--replicate-start", type=int, default=0)
    parser.add_argument("--reference-cell-stop", type=int, default=12,
                        help="Use completed cells 0..N-1, excluding the displayed cell")
    parser.add_argument("--fit-pixels-per-map", type=int, default=128)
    args = parser.parse_args()

    # Validate bounds:
    #   - cell in [0, 39]
    #   - replicate_start in [0, 10] so 5 consecutive replicates stay within [0, 14]
    #   - reference_cell_stop in [2, 40] guarantees at least 1 reference cell even if cell < stop
    #   - fit_pixels_per_map in [16, 2304] since each 48x48 map has exactly 2304 spatial pixels
    if not 0 <= args.cell < SHARDS or not 0 <= args.replicate_start <= 10:
        parser.error("cell must be 0-39 and replicate-start must be 0-10")
    if not 2 <= args.reference_cell_stop <= SHARDS:
        parser.error("reference-cell-stop must be 2-40")
    if not 16 <= args.fit_pixels_per_map <= 2304:
        parser.error("fit-pixels-per-map must be 16-2304")

    # -----------------------------------------------------------------------
    # 2. Load Displayed Target Cell Trajectories (5 Replicates x 6 Stages)
    # -----------------------------------------------------------------------
    samples = read_manifest(args.manifest)
    # Because the manifest cycles through all 40 cells per replicate, replicate `rep`
    # of `args.cell` sits at global manifest index `args.cell + 40 * rep`.
    indices = [args.cell + SHARDS * rep for rep in range(args.replicate_start, args.replicate_start + 5)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Seed PyTorch RNG for reproducible pixel subsampling and randomized PCA iterations.
    torch.manual_seed(42)

    # `maps` accumulates 5 * 6 = 30 tensors in row-major order (replicate, stage),
    # each of shape (C, 48, 48) in float32 on `device`.
    maps: list[torch.Tensor] = []
    rows = []
    for index in indices:
        sample = samples[index]
        cell_dir = args.output_root / f"cell-{args.cell:02d}"
        tensor_path = cell_dir / f"{sample['id']}.safetensors"
        metadata_path = cell_dir / f"{sample['id']}.json"
        final_path = cell_dir / f"{sample['id']}.png"
        if any(not p.is_file() for p in (tensor_path, metadata_path, final_path)):
            raise FileNotFoundError(f"Incomplete packet for {sample['id']} in {cell_dir}")

        # Cross-check saved provenance JSON against manifest and expected Block-20 schema.
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (meta.get("id") != sample["id"] or meta.get("seed") != sample["seed"]
                or meta.get("model_id") != MODEL_ID or meta.get("feature_block") != 20
                or meta.get("feature_percentages") != list(EARLY_PERCENTAGES)):
            raise ValueError(f"Metadata mismatch for {sample['id']}")

        # Load all 6 early feature maps (`block20_000` .. `block20_050`), upcasting from
        # stored bfloat16 to float32 for accurate linear algebra.
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            for percent in EARLY_PERCENTAGES:
                key = f"block20_{percent:03d}"
                tensor = handle.get_tensor(key)
                if tensor.ndim != 3 or tensor.shape[1:] != (48, 48):
                    raise ValueError(f"Unexpected {key} shape in {tensor_path}: {tuple(tensor.shape)}")
                maps.append(tensor.float().to(device))

        # Downsample final 1024x1024 RGB image to <=512x512 in-place to keep HTML size manageable.
        with Image.open(final_path) as im:
            final = im.convert("RGB")
            final.thumbnail((512, 512), Image.Resampling.LANCZOS)
            final_uri = png_uri(final)

        rows.append({"id": sample["id"], "seed": sample["seed"], "final": final_uri, "maps": {}})
        print(f"Loaded {sample['id']}: {sample['prompt']}", flush=True)

    # Verify all loaded feature maps share the exact same channel dimension C (1536 for SD3.5-M).
    channels = maps[0].shape[0]
    if any(m.shape != (channels, 48, 48) for m in maps):
        raise ValueError("Block feature channel counts differ across samples")

    # -----------------------------------------------------------------------
    # 3. Assemble Out-of-Sample Reference Pixel Matrix for PCA Fitting
    # -----------------------------------------------------------------------
    # Exclude the displayed cell so the PCA basis and normalization quantiles are
    # strictly out-of-sample with respect to the visualized cell.
    reference_cells = [cell for cell in range(args.reference_cell_stop) if cell != args.cell]
    if not reference_cells:
        raise ValueError("No independent reference cells available")

    reference_ids = []
    fit_parts = []
    for cell in reference_cells:
        # Use replicates 0..4 (5 images per reference cell).
        for replicate in range(5):
            sample = samples[cell + SHARDS * replicate]
            reference_ids.append(sample["id"])
            tensor_path = args.output_root / f"cell-{cell:02d}" / f"{sample['id']}.safetensors"
            if not tensor_path.is_file():
                raise FileNotFoundError(f"Reference packet missing: {tensor_path}")
            with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
                # Fit PCA only on the earliest three stages (0%, 10%, 20%) so the basis
                # emphasizes early numeracy/layout formation rather than late fine detail.
                for percent in (0, 10, 20):
                    feature = handle.get_tensor(f"block20_{percent:03d}").float().to(device)
                    if feature.shape != (channels, 48, 48):
                        raise ValueError(f"Reference feature shape mismatch in {tensor_path}")
                    # Flatten spatial grid: (C, 48, 48) -> (C, 2304) -> transpose -> (2304, C).
                    pixels = feature.reshape(channels, -1).T
                    # Sample `fit_pixels_per_map` distinct spatial locations without replacement.
                    selection = torch.randperm(pixels.shape[0], device=device)[:args.fit_pixels_per_map]
                    fit_parts.append(pixels[selection])  # shape: (fit_pixels_per_map, C)

    # Hard assertion guaranteeing zero sample overlap between PCA fit set and displayed rows.
    if set(reference_ids) & {row["id"] for row in rows}:
        raise RuntimeError("PCA reference and displayed sample IDs overlap")

    # Concatenate sampled pixel blocks into the design matrix `fit`:
    #   shape = (N_fit, C) where N_fit = len(reference_cells) * 5 * 3 * fit_pixels_per_map
    #   (e.g., 11 * 5 * 3 * 128 = 21,120 rows x 1536 channels by default).
    fit = torch.cat(fit_parts, dim=0)
    mean = fit.mean(dim=0)  # Channel-wise mean vector of shape (C,).
    print(f"Fitting PCA48 from {len(reference_ids)} independent images: "
          f"{fit.shape[0]} pixels × {channels} channels on {device}", flush=True)

    # -----------------------------------------------------------------------
    # 4. Fit Randomized Low-Rank PCA (Top 48 Components) & Canonicalize Signs
    # -----------------------------------------------------------------------
    # Oversample subspace dimension q=56 (> 48) with 3 power iterations for accuracy:
    #   singular_values : (56,)
    #   components (V)  : (C, 56), orthonormal columns
    _, singular_values, components = torch.pca_lowrank(fit, q=56, center=True, niter=3)
    components = components[:, :48]  # Retain top 48 principal axes: shape (C, 48).

    # Resolve SVD sign ambiguity deterministically: enforce that the largest-magnitude
    # channel loading on every component axis is positive.
    for axis in range(48):
        anchor = components[:, axis].abs().argmax()
        if components[anchor, axis] < 0:
            components[:, axis] *= -1

    # Compute cumulative explained variance ratio across components 1..48:
    #   total_variance = ||fit - mean||_F^2 = sum of all squared singular values.
    #   cumulative_variance[k - 1] = sum_{i=0..k-1} s_i^2 / total_variance, shape (48,).
    total_variance = ((fit - mean) ** 2).sum()
    cumulative_variance = (singular_values[:48] ** 2).cumsum(0) / total_variance
    explained3 = float(cumulative_variance[2])
    explained48 = float(cumulative_variance[47])

    # -----------------------------------------------------------------------
    # 5. Calibrate Color & Energy Normalization Scales on Reference Pixels
    # -----------------------------------------------------------------------
    # Project reference pixels into the 48D PCA subspace:
    #   (N_fit, C) @ (C, 48) -> reference_scores of shape (N_fit, 48).
    reference_scores = ((fit - mean) @ components).cpu().numpy()

    # Cumulative L2 norm across PCs 1..k for each reference pixel:
    #   reference_energy[n, k - 1] = sqrt(sum_{j=0..k-1} reference_scores[n, j]^2), shape (N_fit, 48).
    reference_energy = np.sqrt(np.cumsum(reference_scores ** 2, axis=1))

    # Shared global scale: 99th percentile of the full 48-PC norm, allowing direct
    # visual comparison of energy growth as k increases from 3 -> 48.
    shared_energy_scale = float(np.quantile(reference_energy[:, 47], 0.99))

    # 16 cumulative cutoffs: k in {3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48}.
    cutoff_values = range(3, 49, 3)

    # Local per-cutoff scale: 99th percentile of the k-PC norm, keeping small-k energy
    # maps high-contrast even when they capture less total energy than k=48.
    cutoff_energy_scales = {
        k: float(np.quantile(reference_energy[:, k - 1], 0.99)) for k in cutoff_values
    }

    # Per-component 1st and 99th percentile bounds across all 48 axes: shape (48,) each.
    # Indices [:3] are used directly to rescale PC1..PC3 into RGB.
    low = np.quantile(reference_scores, 0.01, axis=0)
    high = np.quantile(reference_scores, 0.99, axis=0)

    # -----------------------------------------------------------------------
    # 6. Project Displayed Cell Features & Render Base64 Map Visualizations
    # -----------------------------------------------------------------------
    # Project all 30 displayed (C, 48, 48) maps into (48, 48, 48) -> (H, W, K) arrays.
    projected = []
    for feature in maps:
        # (C, 48, 48) -> (2304, C) -> center & project -> (2304, 48) -> (48, 48, 48).
        scores = ((feature.reshape(channels, -1).T - mean) @ components).reshape(48, 48, 48)
        projected.append(scores.cpu().numpy())

    for row_index, row in enumerate(rows):
        for stage_index, percent in enumerate(EARLY_PERCENTAGES):
            # Look up flattened (row, stage) projection: shape (48, 48, 48).
            scores = projected[row_index * len(EARLY_PERCENTAGES) + stage_index]
            # Cumulative L2 norm along the component axis (axis=2): shape (48, 48, 48).
            energy = np.sqrt(np.cumsum(scores ** 2, axis=2))

            row["maps"][str(percent)] = {}
            # Render PC1..PC3 as an RGB false-color image using reference [1%, 99%] bounds.
            row["maps"][str(percent)]["rgb"] = png_uri(
                rgb_map(scores[..., :3], low[:3], high[:3]))

            # Render both shared-scale and local-scale cumulative energy heatmaps for each k.
            for k in cutoff_values:
                row["maps"][str(percent)][f"energy_{k}"] = png_uri(
                    energy_map(energy[..., k - 1], shared_energy_scale))
                row["maps"][str(percent)][f"energy_local_{k}"] = png_uri(
                    energy_map(energy[..., k - 1], cutoff_energy_scales[k]))

    # -----------------------------------------------------------------------
    # 7. Serialize Viewer Payload, Export PCA48 Basis Artifacts, & Write HTML
    # -----------------------------------------------------------------------
    first = samples[indices[0]]
    data = {
        "model": MODEL_ID,
        "object_class": first["object_class"],
        "target_count": first["target_count"],
        "stages": list(EARLY_PERCENTAGES),
        "reference_count": len(reference_ids),
        "reference_ids": reference_ids,
        "pca_fit_pixels_per_map": args.fit_pixels_per_map,
        "variance3": round(explained3 * 100, 1),
        "variance48": round(explained48 * 100, 1),
        "cumulative_variance": [round(float(cumulative_variance[k - 1]) * 100, 1)
                                for k in cutoff_values],
        "rows": rows,
    }

    args.html.parent.mkdir(parents=True, exist_ok=True)
    basis_path = args.html.with_suffix(".pca48.safetensors")
    basis_meta_path = args.html.with_suffix(".pca48.json")

    # Save reusable PCA projection and color-calibration tensors:
    #   mean                     : (C,)      float32
    #   components               : (C, 48)   float32
    #   rgb_low                  : (48,)     float32 (full 48-axis 1st percentiles)
    #   rgb_high                 : (48,)     float32 (full 48-axis 99th percentiles)
    #   energy_scale_shared      : (1,)      float32
    #   energy_scales_per_cutoff : (16,)     float32 (ordered for k = 3, 6, ..., 48)
    save_file({
        "mean": mean.detach().cpu().contiguous(),
        "components": components.detach().cpu().contiguous(),
        "rgb_low": torch.from_numpy(low.astype(np.float32)),
        "rgb_high": torch.from_numpy(high.astype(np.float32)),
        "energy_scale_shared": torch.tensor([shared_energy_scale], dtype=torch.float32),
        "energy_scales_per_cutoff": torch.tensor(
            [cutoff_energy_scales[k] for k in cutoff_values], dtype=torch.float32),
    }, str(basis_path))

    # Write machine-readable provenance JSON describing the exact PCA basis & scales.
    basis_meta_path.write_text(json.dumps({
        "model_id": MODEL_ID,
        "feature_block": 20,
        "feature_percentages": list(EARLY_PERCENTAGES),
        "components_kept": 48,
        "reference_ids": reference_ids,
        "fit_pixels_per_map": args.fit_pixels_per_map,
        "fit_percentages": [0, 10, 20],
        "random_seed": 42,
        "explained_variance_first_3": explained3,
        "explained_variance_first_48": explained48,
        "cumulative_explained_variance": {
            str(k): float(cumulative_variance[k - 1]) for k in cutoff_values
        },
        "energy_map": "sqrt(sum of squared PCA scores for components 1..k) at each pixel",
        "energy_scale_shared": shared_energy_scale,
        "energy_scales_per_cutoff": {str(k): cutoff_energy_scales[k] for k in cutoff_values},
        "projection": "(C-channel feature pixel - mean) @ components",
    }, indent=2) + "\n", encoding="utf-8")

    # Inject serialized JSON into the HTML template. Escaping `</` as `<\/` prevents
    # any string containing `</script>` from prematurely closing an inline script block.
    args.html.write_text(HTML.replace("__DATA__", json.dumps(data).replace("</", "<\\/")), encoding="utf-8")
    print(f"Wrote {args.html} ({args.html.stat().st_size / 2**20:.1f} MiB) "
          f"and {basis_path}", flush=True)


if __name__ == "__main__":
    main()