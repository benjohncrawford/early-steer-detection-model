"""Generate SD3.5 Medium numeracy images and early block-20 features.

Overview
--------
This script executes a deterministic, sharded generation pipeline over a 600-sample
numeracy prompt manifest using Stability AI's Stable Diffusion 3.5 Medium
(``stabilityai/stable-diffusion-3.5-medium``).

For each assigned prompt, the script produces three atomically written files under
``<output-root>/cell-<CC>/<SSSS>.*``:
    1. ``<SSSS>.safetensors``:
       - 7 diffusion latent snapshots (``latent_000`` .. ``latent_050``, ``latent_100``),
         each of shape ``(16, 128, 128)`` in ``bfloat16``.
       - 6 intermediate transformer feature maps (``block20_000`` .. ``block20_050``),
         each of shape ``(C, feature_size, feature_size)`` (``(1536, 48, 48)`` for
         SD3.5 Medium with default settings) in ``bfloat16``.
    2. ``<SSSS>.png``:
       - Final 1024x1024 RGB image decoded by the VAE.
    3. ``<SSSS>.json``:
       - Provenance record capturing manifest metadata, generation hyperparameters,
         VAE scale/shift factors, exact scheduler timesteps, tensor shapes, and
         library versions.

Parallelism & Fault Tolerance
-----------------------------
- The 600-row manifest is structured as 15 replicates of 40 canonical
  ``(object_class, target_count)`` cells.
- Work is sharded by ``index % num_shards == shard_id`` where ``num_shards`` divides
  40, ensuring each worker owns a disjoint set of ``cell-XX`` directories with zero
  write contention.
- Existing outputs are validated for configuration consistency, PNG readability, and
  complete ``.safetensors`` key sets before being skipped.
- Writes go to ``*.partial.*`` temporary files first and are promoted via ``os.replace``
  so preempted GPU jobs never leave corrupt final artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Pipeline Constants
# ---------------------------------------------------------------------------

# Hugging Face Hub identifier for Stable Diffusion 3.5 Medium (MMDiT-X, 24 blocks,
# 16 latent channels, patch_size=2, 24 attention heads * 64 dim = 1536 channels).
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"

# Denoising progress checkpoints (%) at which transformer block features are captured.
# Because features are hooked *during* the forward pass of the step starting at that
# progress point, 0% captures the model's very first pass over pure noise and 50%
# captures the pass immediately after 50% of denoising steps have completed.
EARLY_PERCENTAGES = (0, 10, 20, 30, 40, 50)

# Denoising progress checkpoints (%) at which scheduler latents are saved.
# Includes 0% (initial Gaussian noise) through 50% (early trajectory) plus 100%
# (final denoised latent right before VAE decoding).
LATENT_PERCENTAGES = EARLY_PERCENTAGES + (100,)

# Total number of distinct (object_class, target_count) experimental cells:
# 10 object classes x 4 target counts {3, 4, 5, 6} = 40 cells.
SHARDS = 40  # Ten object classes times four requested counts.


def read_manifest(path: Path) -> list[dict]:
    """Load and strictly validate the 600-row numeracy prompt manifest CSV.

    Expected CSV layout
    -------------------
    - Exactly 600 rows ordered as 15 contiguous blocks (replicates 0..14) of 40 rows.
    - Required columns: ``id``, ``prompt``, ``object_class``, ``target_count``,
      ``replicate``, ``seed``.
    - Row ``i`` must satisfy:
        * ``id == f"{i:04d}"``
        * ``replicate == i // 40``
        * ``(object_class, target_count)`` identical to row ``i % 40``
        * ``target_count in {3, 4, 5, 6}``
        * ``seed`` unique across all 600 rows and within signed 64-bit range ``[0, 2**63)``

    Parameters
    ----------
    path : Path
        Filesystem path to the input CSV manifest.

    Returns
    -------
    list[dict]
        Parsed list of 600 validated sample dictionaries, each containing:
        ``{"id": str, "prompt": str, "object_class": str, "target_count": int,
           "replicate": int, "seed": int}``.

    Raises
    ------
    ValueError
        If columns are missing, rows are malformed/out-of-range, seeds collide,
        or the 40-cell cyclic ordering invariant is violated.
    """
    # Use "utf-8-sig" to transparently strip any UTF-8 BOM from spreadsheet exports.
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "prompt", "object_class", "target_count", "replicate", "seed"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest missing columns: {sorted(required - set(reader.fieldnames or []))}")

        samples = []
        for index, row in enumerate(reader):
            # DictReader places overflow columns under key `None` and missing cells as `None`.
            if None in row or any(row.get(field) in (None, "") for field in required):
                raise ValueError(f"Malformed manifest row {index + 2}")

            sample = {
                "id": row["id"],
                "prompt": row["prompt"].strip(),
                "object_class": row["object_class"].strip(),
                "target_count": int(row["target_count"]),
                "replicate": int(row["replicate"]),
                "seed": int(row["seed"]),
            }

            # Validate field-level invariants; seed must fit in signed int64 for torch.Generator.
            if (sample["id"] != f"{index:04d}" or not sample["prompt"]
                    or not sample["object_class"] or sample["target_count"] not in (3, 4, 5, 6)
                    or sample["replicate"] < 0 or not 0 <= sample["seed"] < 2**63):
                raise ValueError(f"Invalid manifest row {index + 2}: {sample}")
            samples.append(sample)

    # Validate global dataset cardinality and seed uniqueness.
    if len(samples) != 600 or len({row["seed"] for row in samples}) != 600:
        raise ValueError("Expected exactly 600 samples with unique seeds")

    # Verify exactly 40 distinct (object_class, target_count) cells exist.
    cells = {(row["object_class"], row["target_count"]) for row in samples}
    if len(cells) != SHARDS:
        raise ValueError(f"Expected {SHARDS} distinct object/count cells, found {len(cells)}")

    # Verify strict cyclic layout so `index % 40` always maps to the same experimental cell
    # and `index // 40` always equals the replicate index.
    for index, sample in enumerate(samples):
        if sample["replicate"] != index // SHARDS:
            raise ValueError(f"Manifest order does not match 40-cell sharding at {sample['id']}")
        if (sample["object_class"], sample["target_count"]) != (
            samples[index % SHARDS]["object_class"], samples[index % SHARDS]["target_count"]
        ):
            raise ValueError(f"Manifest cell order changes at {sample['id']}")

    return samples


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments controlling model sampling, hooks, and sharding."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("prompts/numeracy_smoke_v3.csv"),
        help="Path to the 600-row numeracy CSV manifest.",
    )
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="Root directory where cell-XX subdirectories will be created.",
    )
    parser.add_argument(
        "--steps", type=int, default=40,
        help="Total denoising steps; must be a positive multiple of 10 so 10% checkpoints are exact.",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=4.5,
        help="Classifier-Free Guidance (CFG) scale; must be > 1.0 to enable dual-branch CFG.",
    )
    parser.add_argument(
        "--block", type=int, default=20,
        help="Zero-based MMDiT transformer block index to hook (0..23 for SD3.5 Medium).",
    )
    parser.add_argument(
        "--feature-size", type=int, default=48,
        help="Target spatial height and width for bilinear-resized block features (native grid is 64x64).",
    )
    parser.add_argument(
        "--limit", type=int, default=600,
        help="First N manifest rows; used for a pilot",
    )
    parser.add_argument(
        "--shard-id", type=int, required=True,
        help="Zero-based worker shard index in [0, num_shards).",
    )
    parser.add_argument(
        "--num-shards", type=int, default=SHARDS,
        help="Parallel worker count; must divide 40",
    )
    parser.add_argument(
        "--replicate-start", type=int, default=0,
        help="Inclusive replicate index for checkpointed runs",
    )
    parser.add_argument(
        "--replicate-end", type=int, default=15,
        help="Exclusive replicate index for checkpointed runs",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run and overwrite existing artifacts even if valid files already exist.",
    )
    return parser.parse_args()


def expected_settings(sample: dict, args: argparse.Namespace) -> dict:
    """Build the canonical configuration fingerprint for a single sample.

    Used both to populate the output ``.json`` metadata and to detect parameter
    drift when resuming an interrupted run into an existing ``--output-root``.
    """
    return {
        **sample,
        "model_id": MODEL_ID,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "feature_block": args.block,
        "feature_spatial_size": args.feature_size,
        "feature_percentages": list(EARLY_PERCENTAGES),
        "latent_percentages": list(LATENT_PERCENTAGES),
        "latent_dtype": "torch.bfloat16",
    }


def main() -> None:
    """Run sharded SD3.5 Medium inference with latent and block-feature extraction."""
    # -----------------------------------------------------------------------
    # 1. Argument Validation & Shard Assignment (CPU-only, fast fail)
    # -----------------------------------------------------------------------
    args = parse_args()

    # Percentage checkpoints are multiples of 10% (0, 10, ..., 50, 100), so `steps`
    # must divide cleanly by 10 to land on exact integer step indices.
    if args.steps < 10 or args.steps % 10:
        raise ValueError("--steps must be a multiple of 10")

    # Diffusers only batches [unconditional, conditional] when CFG is active (> 1.0).
    # The forward hook relies on `image_tokens.shape[0] == 2` to slice index 1.
    if args.guidance_scale <= 1:
        raise ValueError("Guidance scale must exceed 1 to capture the conditional CFG branch")

    if args.feature_size < 4 or not 1 <= args.limit <= 600:
        raise ValueError("Invalid feature size or limit")

    # Enforcing `40 % num_shards == 0` guarantees every cell-XX directory belongs
    # exclusively to one shard across all replicates.
    if (args.num_shards < 1 or SHARDS % args.num_shards
            or not 0 <= args.shard_id < args.num_shards):
        raise ValueError("Shard count must divide 40 and shard ID must be in range")

    if not 0 <= args.replicate_start < args.replicate_end <= 15:
        raise ValueError("Replicate range must satisfy 0 <= start < end <= 15")

    samples = read_manifest(args.manifest)

    # Filter manifest to this worker's shard, pilot limit, and replicate window.
    # Each entry is `(global_manifest_index, sample_dict)`.
    assigned = [
        (index, sample) for index, sample in enumerate(samples[:args.limit])
        if index % args.num_shards == args.shard_id
        and args.replicate_start <= sample["replicate"] < args.replicate_end
    ]
    if not assigned:
        print(f"Shard {args.shard_id} has no samples under limit {args.limit}", flush=True)
        return

    # -----------------------------------------------------------------------
    # 2. Lazy Heavy Imports & Pipeline Initialization on GPU
    # -----------------------------------------------------------------------
    # Defer importing PyTorch/Diffusers until validation and sharding succeed so
    # invalid invocations or empty pilot shards exit immediately.
    import torch
    from diffusers import StableDiffusion3Pipeline
    from PIL import Image
    from safetensors import safe_open
    from safetensors.torch import save_file
    from torch.nn import functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run inside a Delta GPU allocation")

    print(f"Loading {MODEL_ID} on {torch.cuda.get_device_name(0)}", flush=True)
    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # Access the MMDiT-X transformer block stack (24 JointTransformerBlock layers for SD3.5-M).
    blocks = pipe.transformer.transformer_blocks
    if not 0 <= args.block < len(blocks):
        raise ValueError(f"Block {args.block} unavailable; model has {len(blocks)} blocks")

    # -----------------------------------------------------------------------
    # 3. Verify Latent & Patch-Token Geometry
    # -----------------------------------------------------------------------
    # For SD3.5 Medium at 1024x1024:
    #   - latent_channels = 16
    #   - vae_scale_factor = 8  -> latent_height = latent_width = 1024 // 8 = 128
    #   - latent_shape = (16, 128, 128)
    #   - patch_size = 2        -> token_height = token_width = 128 // 2 = 64
    #   - flattened image token count = 64 * 64 = 4096 tokens
    latent_channels = int(pipe.transformer.config.in_channels)
    patch = int(pipe.transformer.config.patch_size)
    latent_height = 1024 // pipe.vae_scale_factor
    latent_width = 1024 // pipe.vae_scale_factor
    latent_shape = (latent_channels, latent_height, latent_width)
    if latent_shape != (16, 128, 128) or latent_height % patch or latent_width % patch:
        raise RuntimeError(f"Unexpected model latent/patch configuration: {latent_shape}, patch={patch}")
    token_height, token_width = latent_height // patch, latent_width // patch

    # -----------------------------------------------------------------------
    # 4. Precompute Step-to-Percentage Lookup Maps & Expected Tensor Keys
    # -----------------------------------------------------------------------
    # For default args.steps == 40:
    #   - hook_steps (0-indexed forward call index -> percent completed BEFORE step):
    #       {0: 0, 4: 10, 8: 20, 12: 30, 16: 40, 20: 50}
    #   - latent_steps (1-indexed completed step count -> percent completed AFTER step):
    #       {4: 10, 8: 20, 12: 30, 16: 40, 20: 50, 40: 100}
    #     (0% latent is captured directly before inference starts.)
    hook_steps = {percent * args.steps // 100: percent for percent in EARLY_PERCENTAGES}
    latent_steps = {percent * args.steps // 100: percent for percent in LATENT_PERCENTAGES[1:]}

    # Exact set of 13 keys expected inside each `.safetensors` archive:
    #   7 latents:  latent_000, latent_010, latent_020, latent_030, latent_040, latent_050, latent_100
    #   6 features: block20_000, block20_010, block20_020, block20_030, block20_040, block20_050
    expected_keys = {f"latent_{percent:03d}" for percent in LATENT_PERCENTAGES}
    expected_keys |= {f"block{args.block:02d}_{percent:03d}" for percent in EARLY_PERCENTAGES}

    # -----------------------------------------------------------------------
    # 5. Per-Sample Generation & Feature Extraction Loop
    # -----------------------------------------------------------------------
    for position, (index, sample) in enumerate(assigned, start=1):
        stem = sample["id"]  # Zero-padded 4-digit sample ID, e.g., "0042"
        cell_dir = args.output_root / f"cell-{index % SHARDS:02d}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = cell_dir / f"{stem}.safetensors"
        png_path = cell_dir / f"{stem}.png"
        metadata_path = cell_dir / f"{stem}.json"

        # 5a. Resume & Consistency Guard
        # Prevent silent mixing of incompatible runs in the same output root.
        settings = expected_settings(sample, args)
        if metadata_path.exists():
            old = json.loads(metadata_path.read_text(encoding="utf-8"))
            mismatches = [key for key, value in settings.items() if old.get(key) != value]
            if mismatches and not args.overwrite:
                raise ValueError(f"{stem} settings differ ({', '.join(mismatches)}); use a new output root")

        # Skip sample only if all three artifacts exist, are non-empty, the PNG decodes
        # cleanly, and the safetensors archive contains the exact expected key set.
        if not args.overwrite and all(path.is_file() and path.stat().st_size > 0
                                      for path in (tensor_path, png_path, metadata_path)):
            try:
                with Image.open(png_path) as image:
                    image.verify()
                with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
                    valid = set(handle.keys()) == expected_keys
                if valid:
                    print(f"[{position}/{len(assigned)}] skipping {stem}", flush=True)
                    continue
            except (OSError, ValueError):
                # Corrupt or truncated file; fall through and regenerate cleanly.
                pass

        # 5b. Deterministic Initial Noise Preparation (0% Latent)
        # Seed a CUDA-side generator so initial Gaussian noise `initial` has shape
        # (1, 16, 128, 128) in bfloat16 and is reproducible for this sample's seed.
        generator = torch.Generator(device="cuda").manual_seed(sample["seed"])
        initial = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=latent_channels,
            height=1024,
            width=1024,
            dtype=pipe.transformer.dtype,
            device=torch.device("cuda"),
            generator=generator,
        )
        if tuple(initial.shape) != (1, *latent_shape) or initial.dtype != torch.bfloat16:
            raise RuntimeError(f"Unexpected initial latent: {initial.shape}, {initial.dtype}")

        # Strip batch dim -> store shape (16, 128, 128) on CPU as `latent_000`.
        tensors = {"latent_000": initial[0].detach().cpu().contiguous()}
        snapshots = [{"percent": 0, "completed_steps": 0, "latent_scheduler_timestep": None}]
        feature_timesteps: dict[int, float] = {}
        hook_calls = 0

        # 5c. Forward Hook on Transformer Block (Captures Conditional Image Features)
        def capture_block(_module, _inputs, output):
            """Intercept `blocks[args.block]` output during each denoising forward pass.

            Dimensions through this hook (for SD3.5 Medium, default feature_size=48):
              output              : tuple(text_tokens, image_tokens)
              image_tokens        : (2, 4096, 1536)       [uncond, cond], bfloat16
              image_tokens[1]     : (4096, 1536)          conditional branch only
              .transpose(0, 1)    : (1536, 4096)          channels-first
              .reshape(...)       : (1, 1536, 64, 64)     2D spatial patch grid (NCHW)
              F.interpolate(...)  : (1, 1536, 48, 48)     bilinear-resized, float32
              stored tensor       : (1536, 48, 48)        CHW on CPU, bfloat16, contiguous
            """
            nonlocal hook_calls
            step_index = hook_calls
            hook_calls += 1

            # Ignore steps not in the early percentage schedule.
            if step_index not in hook_steps:
                return

            percent = hook_steps[step_index]

            # JointTransformerBlock returns `(encoder_hidden_states, hidden_states)`,
            # i.e. `(text_tokens, image_tokens)`.
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError(f"Block output is not a (text, image) tuple at step {step_index}")

            image_tokens = output[1]
            # Confirm CFG batch size == 2 and spatial token count == 64 * 64 == 4096.
            if (image_tokens.ndim != 3 or image_tokens.shape[0] != 2
                    or image_tokens.shape[1] != token_height * token_width):
                raise RuntimeError(f"Unexpected block image-token shape: {tuple(image_tokens.shape)}")

            # Diffusers CFG ordering is [unconditional, conditional]. Keep only
            # the prompt-conditioned branch and preserve the spatial token grid.
            channels = image_tokens.shape[2]  # 1536 for SD3.5 Medium
            feature_map = image_tokens[1].transpose(0, 1).reshape(
                1, channels, token_height, token_width
            )

            # Upcast to float32 for accurate bilinear interpolation from (64, 64) -> (48, 48).
            feature_map = F.interpolate(
                feature_map.float(), size=(args.feature_size, args.feature_size),
                mode="bilinear", align_corners=False,
            )

            # Remove dummy batch dim, downcast back to bfloat16, and offload to CPU memory.
            tensors[f"block{args.block:02d}_{percent:03d}"] = (
                feature_map[0].to(torch.bfloat16).detach().cpu().contiguous()
            )
            # Record the exact FlowMatchEulerDiscreteScheduler timestep fed into this step.
            feature_timesteps[percent] = float(pipe.scheduler.timesteps[step_index])

        # 5d. Step-End Callback (Captures Intermediate & Final Scheduler Latents)
        def record_step(_pipeline, step_index, timestep, callback_kwargs):
            """Capture updated latents immediately after the scheduler step finishes.

            `callback_kwargs["latents"]` has shape `(1, 16, 128, 128)` in bfloat16,
            representing the raw scheduler state prior to VAE unscaling/unshifting.
            """
            completed = step_index + 1  # Convert 0-based loop index to 1-based completed count.
            if completed in latent_steps:
                percent = latent_steps[completed]
                # Slice batch index 0 -> shape (16, 128, 128) on CPU.
                tensors[f"latent_{percent:03d}"] = (
                    callback_kwargs["latents"][0].detach().cpu().contiguous()
                )
                snapshots.append({
                    "percent": percent,
                    "completed_steps": completed,
                    "latent_scheduler_timestep": float(timestep),
                })
            return callback_kwargs

        # 5e. Execute Diffusion Sampling with Hook Guaranteed to Clean Up
        hook = blocks[args.block].register_forward_hook(capture_block)
        try:
            with torch.inference_mode():
                # Pass pre-generated `initial` latents so `latent_000` matches step 0 identically.
                image = pipe(
                    prompt=sample["prompt"],
                    height=1024,
                    width=1024,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    latents=initial,
                    generator=generator,
                    callback_on_step_end=record_step,
                    callback_on_step_end_tensor_inputs=["latents"],
                    output_type="pil",
                ).images[0].convert("RGB")  # Guarantees 3-channel (1024, 1024, 3) RGB PIL image.
        finally:
            # Always detach the hook even on exception so hooks never accumulate across samples.
            hook.remove()

        # 5f. Post-Inference Integrity Checks
        # Ensure the hook fired once per step and every expected tensor & snapshot was captured.
        if hook_calls != args.steps or set(tensors) != expected_keys:
            raise RuntimeError(f"Incomplete capture for {stem}: calls={hook_calls}, keys={sorted(tensors)}")
        if len(snapshots) != len(LATENT_PERCENTAGES):
            raise RuntimeError(f"Incomplete latent metadata for {stem}")

        # Guard against NaN/Inf explosions in bfloat16 activations or latents.
        for key, value in tensors.items():
            if not torch.isfinite(value.float()).all():
                raise RuntimeError(f"Non-finite values in {key} for {stem}")

        # Merge feature-capture timesteps into the matching percentage entries in `snapshots`.
        for snapshot in snapshots:
            percent = snapshot["percent"]
            if percent in feature_timesteps:
                snapshot["feature_scheduler_timestep"] = feature_timesteps[percent]

        # 5g. Assemble Provenance Metadata Dictionary
        metadata = {
            **settings,
            "dataset_version": "sd35-numeracy-smoke-v3",
            "manifest_index": index,
            "scheduler": type(pipe.scheduler).__name__,
            "latent_shape_chw": list(latent_shape),  # [16, 128, 128]
            "latent_representation": "scheduler_state_before_vae_scaling",
            # Downstream decoders reconstructing images from saved latents must apply:
            #   vae.decode((latent / scaling_factor) + shift_factor)
            "vae_scaling_factor": float(pipe.vae.config.scaling_factor),
            "vae_shift_factor": float(pipe.vae.config.shift_factor),
            "feature_keys": [f"block{args.block:02d}_{p:03d}" for p in EARLY_PERCENTAGES],
            "feature_shape_chw": list(tensors[f"block{args.block:02d}_000"].shape),  # [1536, 48, 48]
            "feature_representation": "conditional CFG image tokens after transformer block; bilinear resized; no PCA",
            "feature_step_semantics": "block feature before denoising step at given percent completed",
            "snapshots": snapshots,
            "image_shape_hwc": [image.height, image.width, 3],  # [1024, 1024, 3]
            "software": {"torch": torch.__version__, "diffusers": __import__("diffusers").__version__},
        }

        # 5h. Atomic Write via Temporary Partial Files
        # Writing to `*.partial.*` and promoting with `os.replace` guarantees that a crash
        # mid-write cannot leave half-written files under the final target paths.
        tensor_tmp = cell_dir / f"{stem}.partial.safetensors"
        png_tmp = cell_dir / f"{stem}.partial.png"
        metadata_tmp = cell_dir / f"{stem}.partial.json"
        save_file(tensors, str(tensor_tmp))
        image.save(png_tmp, format="PNG")
        metadata_tmp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(tensor_tmp, tensor_path)
        os.replace(png_tmp, png_path)
        os.replace(metadata_tmp, metadata_path)
        print(f"[{position}/{len(assigned)}] saved {stem}: {sample['prompt']}", flush=True)


if __name__ == "__main__":
    main()