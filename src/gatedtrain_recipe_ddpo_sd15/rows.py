"""Registry rows of adapter `ddpo_sd15`: DDPO + LoRA on SD1.5 (area 3).

The training row, the evaluator rows and the adapter class, loaded by `gatedtrain.ext.registry` through the
`gatedtrain.adapters` entry point (MULTIMODAL-SPEC section 6)."""

from __future__ import annotations

from typing import Any

from gatedtrain.ext.rowkit import CONTAINER_PATHS, runtime_dir, venv_runtime_rows

NAME = "ddpo_sd15"
# The runtime directory: this repository's runtime/ (scripts, its .venv, models, prompts, items, constraints; mounted
# read-only at its own host path), or the override; earlier runs lie under the host's output root as
# `{output_root}/<OUT_NAME>`.
RUNTIME_ENV = "GATEDTRAIN_RECIPE_DDPO_SD15_RUNTIME"
OUT_NAME = "ddpo-sd15"
RUNTIME = runtime_dir(RUNTIME_ENV, __file__)
HOST_ROWS = venv_runtime_rows(RUNTIME, OUT_NAME)

TRAINER_ROWS: dict[str, dict[str, Any]] = {}
EVALUATOR_ROWS: dict[str, dict[str, Any]] = {}

# Area "image generation, SD1.5" (MULTIMODAL-SPEC section 6 row 2a with the deviations in runtime/README.md
# "Decisions taken"; runtime/README.md "What gatedtrain needs" items 1-9). The
# trainer is the DDPO port train_ddpo.py on diffusers + peft (decision 1), LoRA cuts on the plain SD1.5 base.
TRAINER_ROWS["ddpo_sd15"] = {
    "uses": ("rl.ddpo@0.1",),
    "pins": ("diffusers", "peft", "torch", "transformers"),          # from env-freeze.txt (MM-33)
    "outputs": ("checkpoint-<epoch>/ LoRA cuts with cut.json written last", "run_args.json", "samples.jsonl",
                "train_steps.jsonl"),
    # EXT-120: train_ddpo.py DEPLOYABLE (cut.json `deployable`): LoRA weights, adapter config, the deployment
    # sampler and the base reference (MM-17, MM-18); trainer_state.json and cut.json are resume state.
    "components": ("pytorch_lora_weights.safetensors", "adapter_config.json", "sampler.json", "base_ref.json"),
    # EXT-123: observed per-sample log (MM-24); a log that does not match the run refuses, there is no fallback.
    "consumption": {"target": "row", "fallback": None},
    "gpu_h_estimate": 1.5,                           # estimate (README "Profiles and cost")
    "evaluators": ("t2i.sd15_webp_items@0.1", "t2i.sd15_jpeg_items@0.1"),   # deciding first (MM-38, EXT-130)
    "framework": "diffusers",
    # EXT-129: the interpreter, the runtime directory (scripts, venv, models, prompts, items, constraints) and the
    # training runs' outputs (an evaluation mounts a cut read-only at /inputs/cut).
    "input_roots": HOST_ROWS["input_roots"],
    # The base and the prompts lie inside the identity mount of the runtime; the registration's sigma comes from
    # the smoke run, so the smoke launch has no spec yet: --spec optional, checked when given (as sb3_ppo).
    "spec_binding": False,
    # README "Request shape": every name, none a credential; PYTORCH_CUDA_ALLOC_CONF is optional (not
    # path-valued; only if the smoke shows fragmentation).
    "env_names": ("HF_HUB_OFFLINE", "HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "PYTORCH_CUDA_ALLOC_CONF",
                  "TOKENIZERS_PARALLELISM", "XDG_CACHE_HOME"),
    "scan_skip": HOST_ROWS["scan_skip"],
    "interpreter_trees": HOST_ROWS["interpreter_trees"],
    "interpreter": HOST_ROWS["interpreter"],
    "container_paths": CONTAINER_PATHS,
    "env_paths": ("HOME", "XDG_CACHE_HOME"),
}

# Area "image generation, SD1.5" (MULTIMODAL-SPEC section 6 rows 2a and 2a-s; runtime/README.md "Decisions
# taken" 2, 3, 5). Scorer: eval_images.py, code_sha256 its SHA-256; the versions are the strings it writes with
# Pillow 13.0.0 (libwebp 1.6.0, libjpeg ABI 6.2), read from its 2026-11-12 base evaluation. An item is one registered
# prompt (an ImageNet animal class held out from training); its `prompt` is the exact-overlap key (EXT-124).
# The SigLIP constraint (`prompt_violation`, t2i.sd15_animal_presence@0.1) is reported only, non-blocking (decision 3),
# and has no row until its constraint-clause wiring: no gate can name it.
# The published eval_images.py differs from the file that produced the 2026-11-13 result
# (sha256:5575c66f09af2bea8cd8ed1c4759bb4982088c58b1811fe616cd99fe7a8972e1) in its docstring (a cleanup, then the
# MIT attribution of its ported code) and in the schema id it writes (`gatedtrain.item_eval` since the rename to
# gatedtrain 0.3.0); only this pin is accepted (RESULT.md).
EVAL_IMAGES_SHA256 = "sha256:504b374e820f5684e5f5e1c4f9bb0584e18edc9a36d8b12996fcca7e709331c7"
_SD15 = {
    "direction": "lower_is_better",
    "score_range": (0, None),
    "binary": False,
    "error_classes": ("scorer_error",),                  # MM-4: the scorer raised (eval_images.py)
    "score_outcomes": ("nonfinite_image",),              # MM-4: a non-finite image is scored (sub_metrics.nonfinite 1)
    "payload_text_field": "prompt",
    "item_payload": ("prompt", "label"),                 # strings; `class_index` (an integer) is not a payload field
    "normalizer": False,
    "env_version": False,
    # EXT-127: the deployment sampler of the cuts (train_ddpo.py / eval_images.py `sampler_doc`), exactly these keys.
    "sampler_keys": ("eta", "guidance_scale", "latents_dtype", "negative_prompt", "noise", "num_steps", "resolution",
                     "safety_checker", "scheduler", "scheduler_config"),
    "framework": "diffusers",
    "samples": "150, 2",
}
EVALUATOR_ROWS["t2i.sd15_webp_items@0.1"] = {
    **_SD15, "metric": "webp_lossless_bpp",
    # 05 section 6.1, R15-1: another engine (libwebp lossless) than the reward's (libjpeg), so T3; it measures the
    # reward's construct (image information content), so it is construct-aligned, and the registration and the
    # report say what a pass shows (decision 2).
    "tier": "T3",
    "construct_aligned": {
        "reward": "jpeg_compressibility",
        "statement": ("webp_lossless_bpp measures the construct of the training reward jpeg_compressibility (image "
                      "information content) with another engine (libwebp lossless, not libjpeg): a pass shows that "
                      "the reward's gain transfers to another codec, not that the images are better (05 section 6.1, "
                      "R15-1)")},
    "scorer": {"name": "webp_lossless_bpp",
               "version": ('eval_images@0.1 pillow 13.0.0 libwebp 1.6.0 {"exact": true, "format": "WEBP", '
                           '"lossless": true, "method": 4, "quality": 100}'),
               "weights_sha256": None, "code_sha256": EVAL_IMAGES_SHA256}}
EVALUATOR_ROWS["t2i.sd15_jpeg_items@0.1"] = {
    **_SD15, "metric": "jpeg_q95_kb", "tier": "T0",     # the training reward itself: supporting, never deciding
    "scorer": {"name": "jpeg_q95_kb", "version": "eval_images@0.1 pillow 13.0.0 libjpeg 6.2 quality 95",
               "weights_sha256": None, "code_sha256": EVAL_IMAGES_SHA256}}

PLUGIN = {"adapters": {"ddpo_sd15": "gatedtrain_recipe_ddpo_sd15.adapter:DdpoSd15Adapter"}, "trainer_rows": TRAINER_ROWS,
          "evaluator_rows": EVALUATOR_ROWS}
