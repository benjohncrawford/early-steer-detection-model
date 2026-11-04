"""Generate SD3 Medium numeracy samples with 11 latent checkpoints and one RGB tensor.

========================================================================================
ARCHITECTURAL & DATAFLOW OVERVIEW
========================================================================================
This script runs text-to-image inference using Stable Diffusion 3 Medium (SD3 Medium)
over a list of text prompts to build a dataset for studying how visual numeracy
(object counts, spatial layouts) emerges across denoising steps.

For each prompt `index` (0, 1, 2, ...), the script produces three files named with a
zero-padded 4-digit stem (`0000`, `0001`, ...):

1. `{stem}.safetensors`: A dictionary of 12 PyTorch tensors saved on CPU:
   - `latent_000`: Initial pure Gaussian noise tensor before any denoising occurs.
     * Shape: `(16, 128, 128)` -> Channels-First `(C, H_latent, W_latent)`
     * Dtype: `torch.float16` (half-precision floating point)
   - `latent_010`, `latent_020`, ..., `latent_100` (10 tensors): Intermediate and final
     denoised latent representations captured at 10%, 20%, ..., 100% of total steps.
     * Shape: `(16, 128, 128)` -> Channels-First `(C, H_latent, W_latent)`
     * Dtype: `torch.float16`
   - `final_rgb`: The final VAE-decoded RGB image converted to a tensor.
     * Shape: `(1024, 1024, 3)` -> Channels-Last `(H_image, W_image, C_rgb)`
     * Dtype: `torch.uint8` (unsigned 8-bit integers in range `[0, 255]`)

2. `{stem}.png`: Standard 24-bit RGB PNG image (`1024x1024` pixels) for visual inspection.

3. `{stem}.json`: Complete provenance metadata (prompt, seed, scheduler class, software
   versions, tensor shapes/dtypes, and per-checkpoint step/timestep logs).

========================================================================================
DIMENSION & LATENT SPACE GEOMETRY IN SD3 MEDIUM
========================================================================================
Unlike older Stable Diffusion models (SD 1.5 and SDXL) which use a 4-channel UNet latent
space, SD3 Medium uses a Multimodal Diffusion Transformer (MMDiT) operating on a
16-channel continuous VAE latent space:
- Target RGB Resolution : `H_image = 1024`, `W_image = 1024`, `C_rgb = 3`
- VAE Spatial Downscale : `pipe.vae_scale_factor = 8` (each 8x8 pixel patch -> 1 latent token)
- Latent Spatial Size   : `H_latent = 1024 // 8 = 128`, `W_latent = 1024 // 8 = 128`
- Latent Channel Count  : `pipe.transformer.config.in_channels = 16`
- Batched Pipeline Shape: `(B, C, H_latent, W_latent) = (1, 16, 128, 128)` on GPU
- Stored Tensor Shape   : `(C, H_latent, W_latent) = (16, 128, 128)` on CPU (batch dim sliced off)
========================================================================================
"""

# Enables postponed evaluation of type annotations (PEP 563). Type hints like `dict[int, int]`
# are stored as strings rather than evaluated at import time, improving startup speed and
# forward-reference compatibility.
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


# Hugging Face Hub repository identifier for Stable Diffusion 3 Medium in `diffusers` format.
MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

# Default path to the input text file containing prompts (one prompt per line).
# - `Path(__file__).resolve()` gets the absolute path of this Python script, resolving symlinks.
# - `.parent` navigates to the directory containing this script.
# - `/ "Prompts" / "numeracy_smoke15.txt"` joins the subdirectory and filename.
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "Prompts" / "numeracy_smoke15.txt"


def checkpoint_steps(num_steps: int) -> dict[int, int]:
    """Map 1-based completed denoising step counts to target decile percentages (10..100).

    Because the total number of inference steps (`num_steps`) may not be a multiple of 10,
    this function computes which completed step index (`1 <= step <= num_steps`) corresponds
    closest to each 10% increment (10%, 20%, ..., 100%) using integer round-half-up arithmetic.

    Mathematical Formulation:
        For each `percent` in `{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}`:
            exact_step = (percent / 100.0) * num_steps
            rounded_step = floor(exact_step + 0.5)
                         = (percent * num_steps + 50) // 100

        Adding 50 before integer floor-division by 100 is equivalent to adding 0.5 before
        truncating, implementing standard round-half-up without floating-point inaccuracies.

    Worked Examples:
        1. `num_steps = 30` (divisible by 10):
           - 10%: `(10 * 30 + 50) // 100 = 350 // 100 = 3`  -> `{3: 10}`
           - 20%: `(20 * 30 + 50) // 100 = 650 // 100 = 6`  -> `{6: 20}`
           - ...
           - 100%: `(100 * 30 + 50) // 100 = 3050 // 100 = 30` -> `{30: 100}`
           Result: `{3: 10, 6: 20, 9: 30, 12: 40, 15: 50, 18: 60, 21: 70, 24: 80, 27: 90, 30: 100}`

        2. `num_steps = 25` (not divisible by 10):
           - 10%: `(10 * 25 + 50) // 100 = 300 // 100 = 3` (exact is 2.5 -> rounds up to 3)
           - 20%: `(20 * 25 + 50) // 100 = 550 // 100 = 5` (exact is 5.0 -> 5)

    Why `num_steps >= 10` is required:
        By the Pigeonhole Principle, if `num_steps < 10`, two different decile percentages
        would map to the same integer step count, overwriting a dictionary key and yielding
        fewer than 10 post-start checkpoints (fewer than 11 total checkpoints including 0%).

    Args:
        num_steps: Total number of denoising steps configured for the scheduler (`>= 10`).

    Returns:
        dict[int, int]: Dictionary where:
            - Key (`int`): 1-based completed step number in `[1, num_steps]`.
            - Value (`int`): Corresponding nominal percentage in `{10, 20, ..., 100}`.
             Note: The 0% checkpoint (`latent_000`) is captured prior to the denoising loop
            and is therefore not included in this lookup table.
    """
    if num_steps < 10:
        raise ValueError("--steps must be at least 10 to yield 11 distinct checkpoints")

    # `range(10, 101, 10)` generates the 10 decile integers: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.
    return {(percent * num_steps + 50) // 100: percent for percent in range(10, 101, 10)}


def parse_args() -> argparse.Namespace:
    """Parse and return command-line interface (CLI) arguments.

    CLI Flags:
        --prompts        : Path to UTF-8 text file with one prompt per line.
        --output-dir     : Directory where `.safetensors`, `.png`, and `.json` files are written.
        --steps          : Number of Euler flow-matching denoising steps (default: 30).
        --guidance-scale : Classifier-Free Guidance (CFG) scale controlling prompt adherence
                           (default: 7.0). Higher values increase prompt alignment at the cost
                           of diversity/saturation.
        --seed           : Base PRNG seed for prompt 0 (`index = 0`). Prompt `index` uses
                           `seed + index` for deterministic, reproducible generation.
        --limit          : Optional cap on how many prompts from the file to process.
        --overwrite      : Boolean flag; if omitted (`False`), existing completed samples are
                           skipped to allow seamless job resumption.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=20261104,
        help="Seed for the first prompt; subsequent prompts use seed + index",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional number of prompts to process")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Execute end-to-end prompt loading, SD3 pipeline initialization, inference, and serialization."""
    # -------------------------------------------------------------------------
    # STEP 1: Fast CLI Validation & Prompt Loading (Before Heavy ML Imports)
    # -------------------------------------------------------------------------
    args = parse_args()

    # Pre-calculate the `{completed_step_count: percent}` lookup dictionary.
    # Validates `args.steps >= 10` immediately before loading weights.
    checkpoints = checkpoint_steps(args.steps)

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if not args.prompts.is_file():
        raise FileNotFoundError(args.prompts)

    # Read the prompt file:
    # - `encoding="utf-8-sig"` decodes UTF-8 and automatically strips any leading Byte Order Mark
    #   (BOM, `\ufeff`) that Windows text editors often insert at the start of text files.
    # - `.splitlines()` splits on `\n`, `\r\n`, or `\r` without retaining newline characters.
    # - `if line.strip()` filters out blank or whitespace-only lines.
    # - `line.strip()` removes leading/trailing spaces from each valid prompt.
    prompts = [
        line.strip()
        for line in args.prompts.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    # Slice the prompt list if `--limit N` was provided.
    # Indexing: `prompts[: args.limit]` selects elements at indices `0, 1, ..., min(len(prompts), limit) - 1`.
    if args.limit is not None:
        prompts = prompts[: args.limit]
    if not prompts:
        raise ValueError("No prompts found")

    # -------------------------------------------------------------------------
    # STEP 2: Lazy Import of Heavy Deep Learning Libraries & GPU Setup
    # -------------------------------------------------------------------------
    # Placed after argument/file validation so `--help` or invalid CLI invocations fail
    # instantaneously without waiting several seconds to import PyTorch, Diffusers, and CUDA.
    import numpy as np
    import torch
    from diffusers import StableDiffusion3Pipeline
    from safetensors.torch import save_file

    # Enforce GPU execution (SD3 Medium has ~2B parameters in its MMDiT + T5-XXL/CLIP text encoders).
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this inside a Delta A100 GPU allocation")

    # Create target output directory (and any missing parent directories) if not already present.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Log the active CUDA device name (index 0 = first visible GPU allocated to the job).
    print(f"Loading {MODEL_ID} on {torch.cuda.get_device_name(0)}", flush=True)

    # Load all SD3 Medium components (3 text encoders: CLIP-L, CLIP-G, T5-XXL; MMDiT transformer;
    # FlowMatchEulerDiscreteScheduler; and 16-channel VAE) in FP16 (`torch.float16`) to halve VRAM usage
    # and utilize Tensor Core acceleration.
    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe.to("cuda")

    # Disable the per-step `tqdm` console progress bar to keep batch/SLURM log files clean.
    pipe.set_progress_bar_config(disable=True)

    # -------------------------------------------------------------------------
    # STEP 3: Verify Latent Space Geometry `(C, H_latent, W_latent)`
    # -------------------------------------------------------------------------
    # Compute expected unbatched 3D latent shape for a 1024x1024 image:
    # - `pipe.transformer.config.in_channels` = 16 latent channels
    # - `pipe.vae_scale_factor` = 8 spatial downsampling factor
    # - `1024 // 8` = 128 latent height and 128 latent width
    # Resulting tuple: `(16, 128, 128)` representing `(Channels, Height, Width)`.
    latent_shape = (
        int(pipe.transformer.config.in_channels),
        1024 // pipe.vae_scale_factor,
        1024 // pipe.vae_scale_factor,
    )
    if latent_shape != (16, 128, 128):
        raise RuntimeError(f"Unexpected latent shape: {latent_shape}")

    steps_to_percent = checkpoints

    # -------------------------------------------------------------------------
    # STEP 4: Main Generation Loop Over Prompts
    # -------------------------------------------------------------------------
    # `enumerate(prompts)` yields `(index, prompt)` where `index` is 0-based (`0, 1, ..., len(prompts)-1`).
    for index, prompt in enumerate(prompts):
        # Format 0-based `index` as a 4-digit zero-padded string (`0 -> "0000"`, `14 -> "0014"`).
        stem = f"{index:04d}"
        tensor_path = args.output_dir / f"{stem}.safetensors"
        png_path = args.output_dir / f"{stem}.png"
        metadata_path = args.output_dir / f"{stem}.json"

        # Checkpoint / Resume Guard:
        # If `--overwrite` is False and all three output artifacts already exist on disk,
        # skip this prompt. Because `.safetensors` is written atomically via `.partial.safetensors`
        # and `metadata_path` is written last, all three existing guarantees the prior run completed.
        if not args.overwrite and all(path.exists() for path in (tensor_path, png_path, metadata_path)):
            print(f"[{index + 1}/{len(prompts)}] skipping {stem}", flush=True)
            continue

        # Derive a unique, deterministic seed for this prompt index.
        seed = args.seed + index
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # ---------------------------------------------------------------------
        # STEP 4A: Sample Initial Gaussian Noise (`latent_000`, 0% Checkpoint)
        # ---------------------------------------------------------------------
        # `pipe.prepare_latents` samples standard normal noise `N(0, I)` on the GPU using `generator`.
        #
        # Dimension & Indexing Details:
        # - `initial` has 4D batched shape `(batch_size, C, H_latent, W_latent)` = `(1, 16, 128, 128)`
        #   in `NCHW` layout, dtype `torch.float16`, residing on `"cuda:0"`.
        initial = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=latent_shape[0],  # Index 0 of `(16, 128, 128)` -> 16 channels
            height=1024,
            width=1024,
            dtype=pipe.transformer.dtype,          # `torch.float16`
            device=torch.device("cuda"),
            generator=generator,
        )

        # Store the 0% checkpoint (`latent_000`) before denoising starts:
        # - `initial[0]` indexes the 0-th element along axis 0 (the batch dimension `B=1`),
        #   reducing the 4D tensor `(1, 16, 128, 128)` to a 3D tensor `(16, 128, 128)`.
        # - `.detach()` strips any autograd computation graph tracking.
        # - `.to("cpu")` copies the tensor from GPU VRAM to host system RAM. Crucially, this
        #   creates an independent copy so subsequent in-place scheduler updates on GPU do not
        #   mutate `latent_000`.
        # - `.contiguous()` guarantees a dense, row-major (C-contiguous) memory layout, which
        #   is strictly required by `safetensors.torch.save_file`.
        tensors = {"latent_000": initial[0].detach().to("cpu").contiguous()}

        # Initialize the chronological list of checkpoint metadata records with the 0% state.
        # Before step 1 begins, 0 steps are completed and no scheduler timestep has been applied yet.
        snapshots = [
            {
                "requested_percent": 0,
                "completed_steps": 0,
                "actual_percent": 0.0,
                "scheduler_timestep": None,
            }
        ]

        # ---------------------------------------------------------------------
        # STEP 4B: Define Per-Step Callback to Capture 10%..100% Latents
        # ---------------------------------------------------------------------
        def record_step(_pipeline, step_index, timestep, callback_kwargs):
            """Hook invoked by `StableDiffusion3Pipeline` at the end of every denoising step.

            Args:
                _pipeline: Reference to the active `StableDiffusion3Pipeline` instance (unused).
                step_index (int): 0-based index of the step that just finished (`0` to `args.steps - 1`).
                timestep (torch.Tensor): 0D scalar tensor representing the continuous flow-matching
                    scheduler timestep (starts at `1000.0` at pure noise and decreases toward `0.0`).
                callback_kwargs (dict): Dictionary containing state tensors requested via
                    `callback_on_step_end_tensor_inputs=["latents"]`.
                    - `callback_kwargs["latents"]` has shape `(1, 16, 128, 128)` on `"cuda:0"`.

            Returns:
                dict: `callback_kwargs` (must be returned so the pipeline loop can continue).
            """
            # Convert 0-based `step_index` (`0 .. steps-1`) to 1-based completed step count (`1 .. steps`).
            completed = step_index + 1

            # Check if this completed step corresponds to one of our 10 target deciles (10%..100%).
            if completed in steps_to_percent:
                percent = steps_to_percent[completed]

                # Format key as `latent_010`, `latent_020`, ..., `latent_100`.
                # Indexing `callback_kwargs["latents"][0]` slices off batch dimension `B=1`:
                # `(1, 16, 128, 128)` on GPU -> `(16, 128, 128)` copied to CPU RAM.
                tensors[f"latent_{percent:03d}"] = (
                    callback_kwargs["latents"][0].detach().to("cpu").contiguous()
                )

                # Record exact step telemetry.
                # `timestep.item()` extracts the scalar value from the 0D PyTorch tensor, and
                # `float(...)` ensures it is a standard Python `float` for JSON serialization.
                snapshots.append(
                    {
                        "requested_percent": percent,
                        "completed_steps": completed,
                        "actual_percent": 100.0 * completed / args.steps,
                        "scheduler_timestep": float(timestep.item()),
                    }
                )
            return callback_kwargs

        # ---------------------------------------------------------------------
        # STEP 4C: Run SD3 Denoising Loop & VAE Decoding
        # ---------------------------------------------------------------------
        # `torch.inference_mode()` disables autograd engine overhead and view tracking,
        # reducing memory footprint and accelerating forward-pass execution.
        with torch.inference_mode():
            # Pipeline execution details:
            # 1. Encodes `prompt` (and an implicit empty negative prompt `""` for CFG) using
            #    CLIP-L, CLIP-G, and T5-XXL.
            # 2. Starts directly from `latents=initial` (shape `(1, 16, 128, 128)`), ensuring
            #    the trajectory begins at the exact `latent_000` tensor we saved above.
            # 3. Runs `args.steps` iterations of the MMDiT transformer + Euler flow-matching step,
            #    calling `record_step` at the end of each iteration (`step_index = 0 .. args.steps-1`).
            # 4. At the final step (`completed == args.steps`), `record_step` captures `latent_100`
            #    (the final denoised latent before VAE decoding).
            # 5. Because `output_type="pil"`, the pipeline then passes `latent_100` through the
            #    SD3 VAE Decoder:
            #      `z_scaled = (latents / vae.config.scaling_factor) + vae.config.shift_factor`
            #    decoding `(1, 16, 128, 128)` -> `(1, 3, 1024, 1024)` in `[-1, 1]`, rescaling to
            #    `[0, 255]`, permuting `NCHW -> NHWC`, and wrapping as a `PIL.Image.Image`.
            # 6. `.images` is a Python `list` of length `batch_size = 1`.
            #    Indexing `.images[0]` extracts the single `PIL.Image.Image` for batch item 0,
            #    and `.convert("RGB")` guarantees a 3-channel 24-bit RGB image.
            image = pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                latents=initial,
                callback_on_step_end=record_step,
                callback_on_step_end_tensor_inputs=["latents"],
                output_type="pil",
            ).images[0].convert("RGB")

        # ---------------------------------------------------------------------
        # STEP 4D: Convert Final RGB Image to Tensor & Validate All Shapes
        # ---------------------------------------------------------------------
        # Convert the `1024x1024` RGB PIL Image into a NumPy array:
        # - Shape: `(1024, 1024, 3)` corresponding to Channels-Last `(Height, Width, Channels)`
        #   where channel axis index 2 is ordered `[Red, Green, Blue]`.
        # - Dtype: `np.uint8` (values `0` through `255`).
        # - `copy=True` allocates a writable standalone buffer owned by NumPy.
        rgb = np.array(image, dtype=np.uint8, copy=True)
        if rgb.shape != (1024, 1024, 3):
            raise RuntimeError(f"Unexpected image shape: {rgb.shape}")

        # Wrap the NumPy array as a CPU PyTorch tensor (`torch.uint8`, shape `(1024, 1024, 3)`).
        # Note the layout distinction inside the `.safetensors` archive:
        # - `latent_000` .. `latent_100`: Channels-First `(C, H, W)` = `(16, 128, 128)`, `float16`
        # - `final_rgb`                 : Channels-Last  `(H, W, C)` = `(1024, 1024, 3)`, `uint8`
        tensors["final_rgb"] = torch.from_numpy(rgb).contiguous()

        # Verify that all 12 expected tensors are present:
        # 1 initial latent (`latent_000`) + 10 decile latents (`latent_010`..`latent_100`) + 1 `final_rgb` = 12.
        if len(tensors) != 12:
            raise RuntimeError(f"Expected 12 tensors, got {list(tensors)}")

        # Explicitly verify the exact shape tuple of every single tensor before writing to disk.
        for key, value in tensors.items():
            expected = (1024, 1024, 3) if key == "final_rgb" else latent_shape
            if tuple(value.shape) != expected:
                raise RuntimeError(f"{key} has shape {tuple(value.shape)}, expected {expected}")

        # ---------------------------------------------------------------------
        # STEP 4E: Assemble Provenance Metadata & Atomically Persist to Disk
        # ---------------------------------------------------------------------
        metadata = {
            "prompt": prompt,
            "prompt_index": index,
            "seed": seed,
            "model_id": MODEL_ID,
            "scheduler": type(pipe.scheduler).__name__,  # Typically "FlowMatchEulerDiscreteScheduler"
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "latent_shape_chw": list(latent_shape),      # `[16, 128, 128]`
            "latent_dtype": str(initial.dtype),          # `"torch.float16"`
            "rgb_shape_hwc": list(rgb.shape),            # `[1024, 1024, 3]`
            "rgb_dtype": "uint8",
            "snapshots": snapshots,                      # List of 11 checkpoint telemetry dicts (0%..100%)
            "software": {
                "torch": torch.__version__,
                "diffusers": __import__("diffusers").__version__,
            },
        }

        # Atomic Write Pattern for `.safetensors`:
        # Each `.safetensors` file is ~9.2 MB (`11 * (16 * 128 * 128 * 2 bytes) + (1024 * 1024 * 3 * 1 byte)`).
        # If the job is preempted or crashes mid-write while writing directly to `{stem}.safetensors`,
        # a truncated/corrupted file would remain on disk.
        # Writing first to `{stem}.partial.safetensors` and then calling `os.replace(temporary, tensor_path)`
        # performs an atomic POSIX filesystem rename once the write is 100% flushed.
        temporary = args.output_dir / f"{stem}.partial.safetensors"
        save_file(tensors, str(temporary))
        os.replace(temporary, tensor_path)

        # Save the visual PNG and JSON metadata sidecar files.
        image.save(png_path)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        # Log 1-based progress (`[1/15]`, `[2/15]`, ...) with `flush=True` for real-time SLURM output.
        print(f"[{index + 1}/{len(prompts)}] saved {stem}: {prompt}", flush=True)


if __name__ == "__main__":
    main()