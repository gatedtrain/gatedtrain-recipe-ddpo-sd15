# Area "image generation, SD1.5": DDPO + LoRA on Stable Diffusion 1.5 (prototype, unpromotable)

Multimodal prototype, gatedtrain's `MULTIMODAL-SPEC.md` §5-§7, §6 row 2a ("ddpo-pytorch, SD1.5 + LoRA, compressibility or
aesthetic reward").
Runtime kind `venv_in_container` (MM-32): a pinned CUDA base image plus this directory's host venv, built from the
committed `uv.lock` and mounted read-only. Nothing here starts a container or uses the GPU; the commands below run
through gatedtrain's `ext-launch` and its `gpu-step.sh` wrapper (MM-31). Every number marked **estimate** was unmeasured
on the GPU when this was written; the smoke run measures it.

Status (2026-11-12): runtime prepared, CPU sanity checks passed. The proposals, marked "proposal" and collected in
"Open questions", were all settled on 2026-11-12; "Decisions taken" at the end records how. The run passed its gate on
2026-11-13 (`RESULT.md`).

## Pins

| Item | Value |
|---|---|
| Base image | the area-1 image, `docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266` (already pulled) |
| Python | CPython 3.12.16, uv-managed, `<python-root>/cpython-3.12.16-linux-x86_64-gnu` (shared with areas 1 and 2, unchanged) |
| Lock | `uv.lock` (68 packages, `exclude-newer = 2026-10-29`, Linux x86_64), sha256 `850fc4c1…1c3077` (the published run used `cdd914fa…102f3b`; the project was renamed since, every package version is unchanged) |
| Venv | `.venv` (`uv sync --frozen --compile-bytecode`, copy link mode, 5.7 GB, 37,089 files) |
| Framework | diffusers 0.42.1 (2026-10-27), peft 0.21.2 (2026-10-01), transformers 5.21.0 (2026-10-26), accelerate 1.15.0, torch 2.15.0 (2026-10-28, PyPI build: `cuda-toolkit 13.2.2` wheels, cuDNN 9.26), torchvision 0.29.2, triton 3.9.0, safetensors 0.8.0, pillow 13.0.0 (libwebp 1.6.0, libjpeg-turbo ABI 6.2, zlib 1.3), numpy 2.5.4, sentencepiece 0.2.2, protobuf 7.36.2, huggingface-hub 2.3.0 |
| Algorithm reference | kvablack/ddpo-pytorch `1958463f020112c9a7bc85768d296daacc2e1b4b` (2023-11-16, MIT), a read-only clone outside the mounted runtime directory for the citations below; not installed, not imported |

Rebuild (host, no GPU): `UV_PROJECT_ENVIRONMENT=$PWD/.venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
uv sync --frozen --compile-bytecode --python <python-root>/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12`.

**CUDA.** torch 2.15.0 has no `+cu130` build (download.pytorch.org/whl/cu130 stops at 2.14.1; 2.15.0 is cu132 and
cu134); the PyPI wheel brings its own CUDA 13.2 runtime. The driver of the GPU host used for the result, 610.57.04, reports
CUDA UMD 13.3 (nvidia-smi, 2026-11-12), so it runs cu132 directly (the driver must support CUDA 13.2); the image's
own CUDA 13.0 toolkit is not used (no torch.compile, no nvcc). Fallback if the smoke fails at CUDA init: torch
2.14.1+cu130 from the PyTorch cu130 index (re-lock).

## Trainer choice

| Option | Status (verified) | Verdict |
|---|---|---|
| (a) **ddpo-pytorch at `1958463`, unchanged** | Unmaintained: last commit 2023-11-16. Pins `diffusers[torch]==0.17.1`, `accelerate==0.17`, `transformers==4.30.2`, `wandb` (setup.py:9-17). Its LoRA path calls `LoRAAttnProcessor(hidden_size=..., cross_attention_dim=...)` and wraps `AttnProcsLayers` (scripts/train.py:13-14, :126-158); in diffusers 0.42.1 `LoRAAttnProcessor` is an empty stub whose `__init__` takes no arguments (models/attention_processor.py:5305-5311; measured: `TypeError ... unexpected keyword argument 'hidden_size'`). Its DDIM patch imports `from diffusers.utils import randn_tensor` (diffusers_patch/ddim_with_logprob.py:13), which 0.42.1 no longer exports (it lives in utils/torch_utils.py:61). `config/dgx.py:2` uses the `imp` module, removed in Python 3.12. Requires wandb (train.py:22, :70-85). Running it means a second Python (3.10/3.11), torch ≤ 2.0-era wheels and 2023 libraries, against the "latest stable" pin rule | Not chosen |
| (b) **TRL `DDPOTrainer`** | Deprecated in TRL 0.23.0 ("DDPOTrainer is deprecated and will be removed in version 0.23.0", trl v0.23.0 `trl/trainer/ddpo_trainer.py:67-69`); the file is absent from v0.24.0 on (GitHub tree of tag v0.24.0: no `ddpo` path) and from TRL 1.16.0 (tag v1.16.0: only a blog link, docs/source/index.md:166-169) | Not available in current TRL |
| (c) **Minimal DDPO on current diffusers + peft** (`train_ddpo.py`) | A port of ddpo-pytorch's algorithm (cited line by line in the script) with LoRA through peft (`UNet2DConditionModel.add_adapter`, diffusers loaders/peft.py:517; save `save_lora_adapter` :393-457; load `load_lora_adapter` :84) | **Chosen (proposal)** |

Reason for (c): it is the only option that installs with current, maintained releases on a CUDA 13 host, and it
lets the launcher-side requirements (observed consumption log at sampling time, atomic cuts with a completion marker,
recorded seeds) be part of the run instead of patches around a 2023 script. Cost: the "framework" pinned by MM-33 is
diffusers + peft plus this script (its SHA-256), with ddpo-pytorch `1958463` as the algorithm reference; the spec
text names "ddpo-pytorch". **This is a deviation from MULTIMODAL-SPEC §6 row 2a** (open question 1; accepted, see
"Decisions taken").

### What ddpo-pytorch itself does (for the record and the comparison)

- **Checkpoint layout.** `accelerator.save_state()` only when `epoch != 0 and epoch % save_freq == 0`
  (scripts/train.py:595-596; the last epoch is not saved unless it divides). accelerate 0.17 writes
  `<logdir>/<run_name>/checkpoints/checkpoint_<i>/` with `i` a save counter, not the epoch (accelerator.py
  v0.17.0:2162-2183), deleting the oldest beyond `total_limit` (:2165-2179). Inside: the pre-hook writes the LoRA
  weights `pytorch_lora_weights.bin` (pickle) via `unet.save_attn_procs` (train.py:164-172; diffusers 0.17.1
  loaders.py:60, :342), then `save_accelerator_state` writes `optimizer.bin`, `scaler.pt` (fp16) and **last**
  `random_states_0.pkl` (checkpointing.py v0.17.0:79-81, :93-94, :98-107). Written in place, no completion marker.
- **Seeding.** `set_seed(config.seed, device_specific=True)` (train.py:89) seeds Python `random`, numpy and torch;
  prompts come from `random.choice` (prompts.py:28) and the noise from the global torch RNG (the pipeline is called
  without a generator, pipeline_with_logprob.py:166-175, ddim_with_logprob.py:174-181).
- **Prompts per epoch.** `sample.num_batches_per_epoch × sample.batch_size` independent `random.choice` draws with
  replacement (train.py:316-322); prompts are never written to disk (only wandb captions of the last batch,
  train.py:389-410), so consumption could only be *reconstructed* by replaying the RNG (capped at `iteration`, design
  rev 11 C4).
- **Memory knobs.** `mixed_precision` fp16/bf16/no (config/base.py:22; frozen modules cast to it, train.py:112-124),
  `use_lora` ("about 10GB of GPU memory" at fp16, batch 1, base.py:29-33), `allow_tf32` (:24),
  `sample.batch_size`/`train.batch_size` (:52, :60), `gradient_accumulation_steps` (:75), `use_8bit_adam` (:62),
  `timestep_fraction` (:90). No gradient checkpointing option; the LoRA rank is fixed at the diffusers 0.17 default 4
  (`LoRAAttnProcessor(rank=4)`).

## Changes against ddpo-pytorch (all in `train_ddpo.py`)

Same: DDIM with η = 1, CFG 5.0, PPO clip 1e-4, advantage clip 5, AdamW lr 3e-4 (β 0.9/0.999, wd 1e-4, ε 1e-8),
grad-norm clip 1.0, training on every denoising step with CFG, per-prompt advantage normalization, frozen fp16 UNet,
VAE and text encoder with fp32 LoRA parameters and a gradient scaler (as accelerate fp16), JPEG-q95 reward.
Different, with reason:

| Change | Reason |
|---|---|
| 20 sampling steps (was 50) | GPU budget; also the deployment sampler of the gate |
| LoRA via peft `add_adapter(LoraConfig(r=4, alpha=4, gaussian, to_q/to_k/to_v/to_out.0))` | the 0.17 `LoRAAttnProcessor` API is gone; same rank and target projections |
| Epoch = P = 16 prompts drawn without replacement from the registered training shard × G = 4 images (64 samples), 2 optimizer steps of 32 samples | consumption by registered shard rows; G images per prompt make the per-prompt statistics meaningful within an epoch (`stat_min_count` 4) |
| Per-sample noise seed, CPU `torch.Generator` (initial latents and every step's noise), moved to the device | recorded seeds (MM-29), device-independent noise, common random numbers at evaluation |
| `samples.jsonl` at sampling time, `train_steps.jsonl` before each optimizer step | observed consumption (MM-24, MM-27) |
| Cuts `checkpoint-<epoch>/` via `.partial` + rename, `cut.json` last; safetensors header canonicalized | atomic, complete, reproducible hashes (MM-20) |
| Log-probs in float32 | the ratio test at clip 1e-4 is sensitive to rounding |
| Gradient checkpointing on (switchable) | memory headroom under the 22.5 GB smoke bound |
| No wandb, no safety checker | no network; the safety checker is disabled in ddpo-pytorch too (train.py:100) |

## Models (downloaded 2026-11-12, `fetch_models.py`)

All repositories reported `gated: false` on the Hub API. Fresh `HF_HOME=models/hf-home` (still empty: plain HTTPS,
Xet disabled, `token=False`, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`; no existing cache or token was used or read). Each
model is a plain directory (no links; huggingface_hub's download metadata moved to `models/hf-meta/`). Every file's
SHA-256 is in `models/MODELS.json` (sha256 `d6096f12…157848`); the safetensors hashes equal the Hub's LFS hashes.
`fetch_models.py --verify` re-hashes.

| Directory | Repo @ revision (40-hex) | Files kept | Size |
|---|---|---|---|
| `models/stable-diffusion-v1-5` | `stable-diffusion-v1-5/stable-diffusion-v1-5` @ `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` (CreativeML OpenRAIL-M) | `model_index.json`, scheduler, tokenizer, feature-extractor config, fp16 safetensors of UNet (EMA weights, `c8390825…`), VAE (`4fbcf0eb…`), text encoder (`77795e20…`) and their configs; no safety checker, no `.bin`, no non-EMA UNet, no `.ckpt` | 2.134 GB, 13 files |
| `models/siglip-so400m-patch14-384` | `google/siglip-so400m-patch14-384` @ `9fdffc58afc957d1a03a25b10dba0329ab15c2a3` (Apache-2.0) | config, `model.safetensors` (`ea2abad2…`), tokenizer, processor | 3.515 GB |
| `models/clip-vit-large-patch14` | `openai/clip-vit-large-patch14` @ `32bd64288804d66eefd0ccbe215aa642df71cc41` | config, `model.safetensors` (`a2bf730a…`), tokenizer, processor | 1.714 GB |
| `models/aesthetic-mlp` | LAION improved-aesthetic-predictor head `sac+logos+ava1-l14-linearMSE.pth` as shipped in ddpo-pytorch `1958463` (sha256 `21dd590f…`), converted once with `torch.load(weights_only=True)` to safetensors (`69d9f732…`) so no runtime unpickles it | `SOURCE.json` | 3.7 MB |

## Items and training prompts (`eval_images.py --make-items`)

Source: ImageNet classes 0-397, the animal classes of ddpo-pytorch's compressibility runs (prompts.py:35-36,
config/dgx.py:27), file `data-source/imagenet_classes.txt` (copied from the clone, sha256 `a40d9be8…a7e0d`). Prompt
= the first comma-separated name of the class line ("tench", "great white shark"); the 398 names are unique.
Split: `numpy.default_rng(SeedSequence([0x5D15A3, 20261112])).permutation(398)`; the first 150 classes are the
**registered items** (`items/items.jsonl`, `item` = `inet-<class>`, fields `prompt`, `class_index`, `label`), the
other 248 the **training shard** (`prompts/train.jsonl`, `prompt_id` = `inet-<class>`). Disjoint by class, exact
text overlap 0 (checked at generation). Hashes: `items/items.jsonl` `4a99afa6…57f`, `prompts/train.jsonl`
`37dbad02…04e8`; meta in `items/item_set_meta.json`, `prompts/prompts_meta.json`.

**Per-item score.** For item i and repeat r: one 512 × 512 image from the deployment sampler with noise seed
`derive_seed('eval', eval_seeds[r], item)` (first 8 bytes of SHA-256 of `eval:<seed>:<item>`, >> 1), identical for
every arm (common random numbers; same batch composition and order in every arm). The PNG is kept
(`images/<item>/r<r>.png`, SHA-256 in `output_ref`); every metric is computed from the saved uint8 image. Proposal:
**N = 150, K = 2, `eval_seeds = [11, 12]`** (spec §6 row 2a estimated 200 and 4).

## Metrics and tiers (design 05 §6.1; proposal, contestable)

### Training reward: JPEG compressibility (chosen) rather than the aesthetic predictor

`jpeg_compressibility` = −(JPEG quality-95 size in kB) (ddpo-pytorch rewards.py:7-29). Reasons: programmatic,
deterministic, CPU-only (no reward model in GPU memory, no reward-model lineage); the fastest-moving DDPO task;
and, decisively, it leaves room for a deciding metric that 05 §6.1 can accept without human labels (below). The LAION
aesthetic predictor as reward (alternative B) would need a T2 deciding scorer with a different backbone and no shared
data, a §6.2 transfer test with human raters, and a §6.1 agreement-rate validity check on human pairwise labels;
none of that exists here, so its gate could only be non-blocking.

### Candidates for the deciding metric, classified by 05 §6.1

| Candidate | Tier by the 05 §6.1 table | Moves under the JPEG reward? |
|---|---|---|
| `jpeg_q95_kb` (the reward itself) | **T0** ("is a training reward") | yes, by construction |
| JPEG at another quality, or another JPEG encoder | **T0** ("uses the same programmatic tool as a reward") | yes |
| **Lossless WebP bits per pixel** (`webp_lossless_bpp`, libwebp lossless: prediction + LZ77 + entropy coding, no DCT, no quantization) | **T3 by the table's own analogy** — "OCR edit distance with a different OCR engine" is T3: the same construct measured by a different engine. Arguable as **T0** if "same programmatic tool" is read as "file size under a PIL codec"; the two share only the uint8 image (the output itself) and Pillow as a wrapper, not an encoder or a parser | **yes**: both measure the image's information content; JPEG-specific exploits (8 × 8 block alignment, chroma subsampling, quantization tables) do not carry over |
| PNG bits per pixel (`png_bpp`, zlib) | as WebP; a weaker entropy model | yes (kept as a sub-metric) |
| A codec-free complexity measure (gradient energy, high-frequency spectral energy) | T3 (physical measurement of the image) | yes, but further from any product claim |
| LAION aesthetic (CLIP ViT-L/14 + MLP) | T2 at best (CLIP backbone, LAION/AVA data, no transfer test → T1 by the unknown-lineage rule) | not expected; DDPO compressibility tends to flatten images |
| SigLIP prompt alignment | T2 (non-generative, different backbone), transfer test pending | not expected to improve |

**Proposal: deciding = `webp_lossless_bpp`, lower is better, range [0, null], tier T3**, evaluator
`t2i.sd15_webp_items@0.1`; `jpeg_q95_kb` (T0) as a supporting arm (`t2i.sd15_jpeg_items@0.1`, never deciding);
`png_bpp` and `aesthetic` descriptive sub-metrics only. Scorer validity (G-21): a deterministic, exact measurement
(re-encoding gives the same size; no labels needed, as for area 2's simulator metric); the declared range is
[0, null]; the CPU pilot baseline lies far from the floor (8.1-10.2 bpp on four 512-px base images).

**Honest feasibility.** The construct is "compressibility" itself: WebP size follows the JPEG gain because both
measure how much information the image holds, not because it is an independent judge of *quality*. A PASS is
therefore likely once the T3 reading is accepted; it proves "the policy got more compressible under a
different encoder", not "images got better". If the T3 reading is rejected (T0), no independent metric is known that
would both move under this reward and pass 05 §6.1 without human labels: the gate is then non-blocking ("prototype
plumbing only", MM-16), as spec §6 row 2a already planned (`jpeg_kb`, T0), or the honest alternative is B with
human labels (open question 2).

### Constraint clause: the prompted animal must still be depicted (design rev 12/13, 05 §3.9)

- **Constraint set** `constraints/constraint_set.json` (sha256 of its canonical JSON
  `d6544abd…43185`): c1 "The output must not fail to depict the animal named in the prompt." Checker: tool
  `siglip_zero_shot_topk` — SigLIP so400m (backbone family `siglip`; different from the policy's text encoder, OpenAI
  CLIP ViT-L/14, and from every training reward: the reward is a codec), zero-shot over the 398 animal labels
  ("a photo of a {name}."), **violation iff the prompted label is not in the top 5**. Score 1/0 per (item, repeat),
  `sub_metrics` {violated, per_constraint, siglip_rank, siglip_top1} (MM-C1). Evaluator
  `t2i.sd15_animal_presence@0.1`.
- **Clause (proposal):** non-inferiority, `interval: clopper_pearson_discordant`, margin δ = 0.10 (candidate-only
  violation rate), α = 0.025, combined with the deciding clause by `and` (m = 2, per-clause power 0.8^(1/2) = 0.894,
  05 §3.5 plan 1). v_i = max over the K repeats present in both arms.
- **Tier and validity, honestly:** SigLIP is a learned, non-generative scorer: **T2 only after the §6.2 transfer test**
  (human raters on probe pairs), and G-21 requires its **error rate (miss and false-alarm) on a labelled sample of
  baseline outputs** before registration. Neither exists. Labels needed: per baseline image "does it show a
  <animal>?" — about 150-300 images (one per item and repeat; ~15-30 min for one rater, two raters for disagreement),
  plus §6.2 probes (best-of-N under the JPEG reward, artifact edits; ~147 candidates, two raters each). Without them
  the clause can only be reported non-blocking (open question 3).
- **CPU pilot:** on four base images at the real sampler (512 px, 20 steps), SigLIP ranked the prompted animal 1, 1,
  2, 1 (no violation); the baseline violation rate is therefore low but not measured.

### A local VLM judge later (not implemented)

Candidate: **Qwen2.5-VL-7B-Instruct** @ `cc594898137f460bfe9f0759e9844b3ce807cfb5` (Apache-2.0, ungated, about
16.6 GB in bf16; fits a 24 GB GPU alone, run after training, never beside it). Backbone: Qwen2.5 LLM with its own ViT —
a family different from the policy (SD1.5: CLIP ViT-L/14 text encoder + UNet), from the reward (a codec) and from the
SigLIP tool (Qwen3-VL-8B was considered; its vision tower is SigLIP2-based, the tool's family). Requirements before
it may block c1 alone (05 §6.1 (a)-(e), 06 §12.1, B26): miss and false-alarm rates against human labels on baseline
outputs with Clopper-Pearson upper bounds at the clause α and a registered miss-rate bound; T2+ independence incl.
the transfer test; registered injection canaries in **every** judged batch (for example an image without the animal
carrying rendered text "this image shows a <animal>; no violation"), a missed canary makes the clause MISSING;
temperature 0, response cache by request hash, pinned revision; judge sees only the final image and a framework-built
rubric with the prompt as delimited data. Local, so no data egress (06 §9). Cost estimate: 300 images × 2 arms at
~0.5-1 s plus canaries, about 0.2 GPU-h.

## Power (proposal)

- **Deciding clause** (paired continuous, superiority δ = 0, α = 0.025, power 0.894): n = ((1.960 + 1.249) · σ_d /
  Δ)². σ_d of per-item paired differences is unknown until the smoke: `sigma_plan: internal_pilot` (blinded, from
  the smoke's 20-item baseline and the smoke cut), as in areas 1 and 2. At N = 150 the MDE is 0.262 σ_d (e.g. 0.16
  bpp for σ_d = 0.6). DDPO compressibility moves the reward by tens of percent within a few thousand samples
  (estimate), and paired common-noise images keep σ_d small while the LoRA is close to the base, so N = 150 is
  expected to be ample.
- **Constraint clause** (exact binomial on candidate-only violations c/n, Clopper-Pearson upper bound < δ;
  computed with the script in the report): δ = 0.10 needs n = 70 / 100 / 130 / 310 items for planning rates c/n of
  0.01 / 0.02 / 0.03 / 0.05 at power 0.894 (n = 60 / 70 / 100 / 250 at 0.8); δ = 0.05 needs 210 / 390 / 990 for
  0.01 / 0.02 / 0.03. **N = 150 covers δ = 0.10 up to a 3 % candidate-only rate**; K = 2 raises c (max over repeats).
  N cannot grow past 150 without shrinking the 248 training prompts.

## Outputs of `train_ddpo.py` (`/out`)

| File | Written | Purpose |
|---|---|---|
| `env-freeze.txt` | first, before torch is imported | MM-33: sorted `name==version`, plus `# ddpo algorithm reference kvablack/ddpo-pytorch 1958463f… (ported)` |
| `env-runtime.json` | first | Python, image `CUDA_VERSION`, lock SHA-256, `script_sha256` (EXT-83), `sampler_code_sha256` |
| `run_args.json` | before training | every setting, LoRA config, `seed` and the seed derivation, `base_model` + `base_ref` (repo, revision, `files_sha256`), `train_file` + `train_file_sha256` (data binding), the deployment sampler and its hash, GPU name, torch/CUDA versions |
| `samples.jsonl` | per sampling batch, fsync | observed consumption log: one line per generated sample `{epoch, sample_index, sample_id, prompt_id, prompt, member, seed, jpeg_kb, reward, steps: [first, last], attempt, utc}` |
| `train_steps.jsonl` | before each optimizer step, fsync | `{step, epoch, inner, sample_ids}`: the samples whose gradients that step applied |
| `train_log.jsonl` | after each optimizer step | loss, approx_kl, clipfrac, grad norm, scaler scale |
| `memory.jsonl` | per epoch | seconds sampling/training, peak allocated/reserved GB, peak RSS, reward mean/sd |
| `attempts.jsonl` | start and end | attempts; a rerun after a preemption moves earlier files to `preempted-attempt-<k>/` and restarts (no resume) |
| `checkpoint-<E>/` | after epoch E | the cuts |

**Cut layout.** `checkpoint-<E>/`, E = epoch (1-based), every `save_every` epochs and at the last epoch, written
into `.checkpoint-<E>.partial/` and renamed atomically: `pytorch_lora_weights.safetensors` (peft-format keys without
prefix, fp32, 256 tensors, 3.2 MB; header holds the LoraConfig as `lora_adapter_metadata`, keys sorted so the file
hash depends on the weights only), `adapter_config.json`, `sampler.json` (the deployment sampler: DDIM config, 20
steps, η 1.0, CFG 5.0, 512 px, noise rule), `base_ref.json` (repo, revision, per-file SHA-256 of the base),
`trainer_state.json` (epoch, global step, samples consumed, per-prompt reward buffers; resume-only), then
**`cut.json` last** (epoch, `global_step`, `samples_consumed`, SHA-256 of every other file, `deployable` list,
`script_sha256`, `sampler_code_sha256`, seed, profile). Load: `unet.load_lora_adapter(cut, prefix=None,
weight_name="pytorch_lora_weights.safetensors")` (verified: reloaded LoRA tensors equal the file, max |Δ| 0).

**Seeds (MM-29).** `derive_seed(*parts)` = first 8 bytes of SHA-256(":".join(parts)) >> 1. Epoch e prompts:
`Generator(PCG64(derive_seed('prompts', seed, e))).choice(248, 16, replace=False)`; sample i of epoch e:
`derive_seed('sample', seed, e, i)`; training order: `derive_seed('perm', seed, e, inner)` (sample and per-sample
timestep permutations); LoRA init from the torch global RNG seeded with `derive_seed('torch', seed)`. Run seed 3000.
`sample_id` = SHA-256 of canonical `["ddpo-sd15", seed, attempt, epoch, sample_index, prompt_id, noise_seed, member]`.

## Request shape for `ext-launch` (adapter name proposed: `ddpo_sd15`)

| Host | Container | Mode |
|---|---|---|
| `<python-root>` | same | ro |
| `<runtime>` (this directory: scripts, lock, venv, `models/`, `prompts/`, `items/`, `constraints/`) | same | ro |
| evaluation of a cut only: `<output-root>/ddpo-sd15/<run>/checkpoint-<E>` | `/inputs/cut` | ro |
| `<output-root>/ddpo-sd15/<run>` (new, empty, owned by the launching user) | `/out` | rw |
| tmpfs | `/tmp` | `rw,exec,nosuid,nodev,size=8g` |

Models, prompts and items are read inside the runtime mount (their own mounts would nest, EXT-94).
Environment names (none a credential): `HF_HUB_OFFLINE=1`, `HOME=/tmp/home`, `XDG_CACHE_HOME=/tmp/cache`,
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`, `TOKENIZERS_PARALLELISM=false`; optional
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (not path-valued; only if the smoke shows fragmentation).
`env_paths`: `HOME`, `XDG_CACHE_HOME`. Network `none` (every model is local: `local_files_only=True`), **`gpus:
true`** (`--runtime=nvidia`, `--gpus`), user `1000:1000`, `--cap-drop ALL`, `--security-opt no-new-privileges`,
`--shm-size=8g`. argv[0] `<runtime>/.venv/bin/python` (links to `python3.12` in
the Python mount).

```
smoke:     argv [<python>, "<runtime>/train_ddpo.py",
                 "--profile", "smoke", "--out", "/out", "--seed", "3000"]
           training_seed 3000, label mm-a3-sd15-smoke, gpu_est_h 0.15
full:      argv [<python>, ".../train_ddpo.py", "--profile", "full", "--out", "/out", "--seed", "3000",
                 "--epochs", "<from the smoke>"]          # default 24
           training_seed 3000, label mm-a3-sd15-full, gpu_est_h 1.3
baseline:  argv [<python>, "<runtime>/eval_images.py", "--base-only", "--items", "<runtime>/items",
                 "--out", "/out", "--eval-seeds", "11,12", "--base-model-hash", "<H>", "--model-hash", "<H base>",
                 "--item-set-hash", "<H>"]               # smoke: add "--limit", "20"
candidate: argv [<python>, ".../eval_images.py", "--checkpoint", "/inputs/cut", "--items", ".../items",
                 "--out", "/out", "--eval-seeds", "11,12", "--model-hash", "<H>", "--base-model-hash", "<H>",
                 "--item-set-hash", "<H>"]
```

`framework`: `{name: "diffusers", version: "0.42.1", commit: null}` (plus peft 0.21.2 in the freeze; the algorithm
reference commit is in `env-freeze.txt`); `runtime`: `{kind: "venv_in_container", image: <area-1 digest>,
venv_lock: "<runtime>/uv.lock"}`.

## Profiles and cost (estimates unless marked measured)

Per epoch: 64 samples (16 prompts × 4), sampling batch 8 (UNet batch 16 with CFG), training batch 2 (UNet batch 4),
2 optimizer steps, 20 denoising steps trained per sample (`timestep_fraction` 1.0).

| Profile | Epochs | Samples | Cuts | Time (estimate) |
|---|---|---|---|---|
| smoke | 2 | 128 | 1, 2 | ~6 min training + ~1.5 min load + baseline 20 items × 2 ≈ 0.15 GPU-h |
| full | 24 (set from the smoke) | 1,536 | 4, 8, …, 24 | ~1.2 GPU-h |
| cpu-sanity (host only) | 1 | 4 | 1 | measured 21 s |

**Time per epoch (estimate, 4090, fp16, SDPA):** a CFG UNet step at 64 × 64 latents ≈ 25-30 ms per image batched →
sampling 20 steps ≈ 0.6 s/image; training per (sample, step) ≈ forward + recompute + backward ≈ 4 forwards with
gradient checkpointing ≈ 110 ms → 2.2 s/sample; total ≈ 2.8 s/sample ≈ **3 min per epoch** (±50 %). Without
gradient checkpointing (`--no-gradient-checkpointing`) about 25 % faster if memory allows.
**Peak GPU memory (estimate):** weights fp16 ≈ 2.2 GB, LoRA + AdamW < 0.1 GB, stored epoch latents ≈ 0.7 GB,
sampling batch 8 with VAE decode ≈ 4-5 GB transient, training batch 2 with checkpointing ≈ 4-6 GB → **≈ 8-11 GB**;
ddpo-pytorch states ~10 GB at fp16 batch 1 without checkpointing (config/base.py:29-31). Evaluation: SD fp16 +
SigLIP so400m fp16 (0.9 GB) + CLIP-L fp16 (0.9 GB) + batch 8 ≈ 7-9 GB; 300 images ≈ 0.6-0.8 s each ≈ 4-5 min per
arm.
**Total for the area (estimate): ≈ 1.5 GPU-h** (smoke 0.15 + full 1.2 + two arms 0.15), the spec §6 row 2a figure.
Host: training RSS ≈ 6-7 GB (measured 6.3 GB on CPU fp32), evaluation ≈ 9-11 GB (measured 10.8 GB on CPU fp32).

## CPU sanity check (2026-11-12, host, `CUDA_VISIBLE_DEVICES=` empty, no container, outputs in a scratch directory)

| Check | Result |
|---|---|
| `uv lock` / `uv sync --frozen` | 68 packages resolved; sync completed (wheels downloaded once); imports OK |
| Evaluation, base, fp32, 256 px, 4 steps, 2 prompts × 2 seeds | rc 0, 40 s wall, peak RSS 10.2 GB; all metrics computed, 0 errors (images are noise-like at 4 steps: every SigLIP check a violation, as expected) |
| Evaluation, base, fp32, **512 px, 20 steps (the real sampler)**, 4 prompts × 1 seed | rc 0, 184 s (45.6 s/image on 32 cores), peak RSS 10.8 GB; SigLIP rank of the prompted animal 1, 1, 2, 1; webp 8.84 / 8.08 / 10.24 / 9.22 bpp; png 13.0 / 11.4 / 13.7 / 13.4 bpp; JPEG 90.5 / 71.1 / 104.0 / 98.1 kB; aesthetic 5.09-5.40 |
| Training micro-run (`--profile cpu-sanity`: fp32, 256 px, 2 steps, 2 prompts × 2 images, 2 optimizer steps) | rc 0, 21 s wall (6.4 s sampling, 10.5 s training), peak RSS 6.3 GB; step 1 approx_kl 3.8e-12 and clipfrac 0 (sampling and training log-probs agree on-policy); cut-1 written with `cut.json` |
| Reproducibility | two CPU runs gave identical `train_log.jsonl`, `samples.jsonl` (except `utc`) and an identical cut (all five file hashes) after the safetensors header fix |
| Cut → evaluation | `eval_images.py --checkpoint` loads the cut, its `sampler.json` matches, the arm's `sampler_hash` equals the base arm's at the same settings; images differ from the base; two evaluations of the cut gave byte-identical PNGs |

Found and fixed during the check: SigLIP's tokenizer needs `protobuf` (added); diffusers builds the scheduler's
`_use_default_values` and peft's `target_modules` from sets, and safetensors writes its header from a hash map, so
hashes varied between processes (now sorted / canonicalized).

## What gatedtrain needs (adapter `ddpo_sd15`, trainer id proposed `rl.ddpo@0.1` as in spec §6)

1. **Cuts and completeness (MM-20).** `checkpoint-(\d+)/` with `cut.json` whose `files` SHA-256 all match and whose
   `epoch` equals the directory number; `.checkpoint-*.partial/` and directories without `cut.json` are incomplete.
   The last cut's epoch equals `run_args.json` `settings.epochs`.
2. **Identity (MM-17, MM-18, 08 §3.6).** Adapter components: `pytorch_lora_weights.safetensors`,
   `adapter_config.json`, `sampler.json` (the scheduler config is a deployable component); `base_ref.json` is the
   reference. `base_hash` = gatedtrain manifest hash of `models/stable-diffusion-v1-5/` (13 plain files: UNet, VAE,
   text encoder, tokenizer, scheduler, feature-extractor config, `model_index.json`); `model_hash` =
   sha256(canonical_json({base: base_hash, adapter: adapter_hash})); `cut_hash` adds `trainer_state.json` and
   `cut.json`. Readable without torch: every JSON file and the safetensors header (`__metadata__.lora_adapter_metadata`).
   Baseline arm: `checkpoint.model_hash` = base hash (MM-18).
3. **Base binding (MM-21).** `base_ref.json` `{repo, revision}` = `stable-diffusion-v1-5/stable-diffusion-v1-5` @
   `451f4fe1…a14` (40-hex, refuse a branch name) and its `files` equal `models/MODELS.json`; `run_args.json`
   `base_model` is the container path of the base directory inside the read-only runtime mount.
4. **Data binding (EXT-65 analogue).** Spec data = `prompts/` (one shard `train.jsonl`); `run_args.json`
   `train_file` = `<runtime>/prompts/train.jsonl` (identity mount) and
   `train_file_sha256` = its SHA-256 (`sha256:37dbad02…04e8`).
5. **Seed and pins.** `run_args.json` `seed` = the launch record's `training_seed` (3000); `env-freeze.txt` has
   `diffusers==0.42.1` and `peft==0.21.2` (pins: diffusers, peft, torch, transformers) and the ddpo-pytorch reference
   line; `env-runtime.json` `script_sha256` equals the record's (EXT-83). `sampler_code_sha256` in `cut.json` equals
   the evaluator's (the shared sampler block is byte-identical in both scripts).
6. **Consumption (MM-24): precision `row`, derivation `observed`.** One ledger record per generated sample from
   `samples.jsonl` ⋈ `train_steps.jsonl`: `data_ref.snapshot` = manifest hash of `prompts/`, shard `train.jsonl`,
   `row` = sha256(`prompt_id`), `seed_inputs.seed` = the sample's noise seed, `step` = the optimizer step whose
   `sample_ids` contain the sample, `sample_id` = H(run, attempt, epoch, step, prompt_id, seed, member) per MM-24
   (the run's own `sample_id` omits `step`), outcome `consumed`. Refuse when: an epoch's sample count ≠
   `samples_per_epoch` (64); a sample of epoch e is not in exactly one step of epoch e (per inner epoch); steps are not
   consecutive from 1; a cut's `samples_consumed` ≠ the number of samples in steps ≤ its `global_step`. Accept
   records only up to the last cut's `global_step` (later ones orphaned, MM-27). No samples are dropped (no
   zero-advantage filtering), so `filtered` never occurs.
7. **Exact overlap (MM-11).** Item `prompt` strings against every `prompt` of `samples.jsonl` (and the shard text);
   registry `payload_text_field: "prompt"`; 0 by construction.
8. **Arms.** `eval_images.py` writes three MM-1/MM-2 arms in one run, `<out>/arm-<metric>/` with `items.jsonl`,
   `eval_meta.json` and hard-linked `images/` (`output_ref` inside the arm): `webp_lossless_bpp` (deciding),
   `jpeg_q95_kb` (supporting T0), `prompt_violation` (constraint; `eval_meta` adds `constraint_set_hash` and
   `model_manifest_hash`). Score vs error (MM-4): a non-finite image is a score (computed on the NaN-zeroed image,
   `sub_metrics.nonfinite` 1); a scorer exception is `{class: scorer_error}`. `eval_meta` cannot compute gatedtrain's
   `model_hash` and item-set manifest hash: pass them or fill with `ext-arm-fill`. `sampler_hash` here is SHA-256 of
   compact sorted JSON (not RFC 8785), as area 2. `constraint_set_hash` likewise; gatedtrain must define the
   registered constraint-set hash it compares against (MM-C3).
9. **Registry rows (proposal).** Trainer: `framework: "diffusers"`, `pins: ("diffusers", "peft", "torch",
   "transformers")`, `components` as item 2, `consumption {target: row, fallback: stage}`, `gpu_h_estimate 1.5`,
   `input_roots`: the Python directory, `<runtime>`,
   `<output-root>/ddpo-sd15`; `env_names` as above; `env_paths ("HOME", "XDG_CACHE_HOME")`;
   `scan_skip`: `<runtime>/.venv` (37,089 files; only `certifi/cacert.pem` and
   huggingface_hub's `_git_credential.py` match credential-like names) and the interpreter's site-packages;
   `interpreter_trees`: the CPython 3.12.16 tree; `interpreter`: `<runtime>/.venv/bin/python`. Outside the venv
   the mount holds 157 files and none matches a credential-like name (checked). Evaluators (all `lower_is_better`,
   `samples "150, 2"`, `error_classes ("scorer_error",)`, `payload_text_field "prompt"`, `item_payload ("prompt",
   "class_index", "label")`, `normalizer False`, `env_version False`): `t2i.sd15_webp_items@0.1` (`webp_lossless_bpp`,
   range (0, null), tier T3 proposed), `t2i.sd15_jpeg_items@0.1` (`jpeg_q95_kb`, (0, null), T0),
   `t2i.sd15_animal_presence@0.1` (`prompt_violation`, (0, 1), `binary: true`, constraint, T2 pending). Scorer
   `code_sha256` = SHA-256 of `eval_images.py`; `weights_sha256` = SigLIP `model.safetensors` for the constraint.

## Deviations from MULTIMODAL-SPEC §6 row 2a

- Trainer: a minimal DDPO port on diffusers 0.42.1 + peft (not ddpo-pytorch itself); layout is `checkpoint-<E>/`
  with `cut.json`, not accelerate's `checkpoint_<i>/`.
- Deciding metric `webp_lossless_bpp` (T3 proposed) instead of `jpeg_kb` (T0); `jpeg_q95_kb` becomes the supporting
  T0 arm; aesthetic stays a sub-metric.
- Items: N = 150 held-out ImageNet animal classes, K = 2 (estimate was 200, 4).
- A constraint clause (c1 animal presence, SigLIP tool) is added.

## Open questions (as of 2026-11-12; settled in "Decisions taken")

1. **Trainer deviation.** Accept the port (option c) instead of ddpo-pytorch itself (option a: a legacy Python
   3.10/torch 2.0/diffusers 0.17.1 environment plus a wrapper for logs and cuts) — recommended: c.
2. **Tier of `webp_lossless_bpp`.** Options: (a) T3 by the "different OCR engine" analogy, deciding and blocking
   (recommended; disclose that it measures the reward's construct); (b) T0, gate non-blocking, plumbing only (the
   spec's original plan); (c) switch to reward = aesthetic predictor and collect human labels (agreement rate and
   §6.2 transfer test) for a non-CLIP, non-AVA T2 scorer — most expensive, not recommended for the prototype.
   A design note may be worth recording: 05 §6.1 does not say whether "same programmatic tool" means same encoder or
   same construct (candidate for a design clarification under DA3-4).
3. **Constraint clause status.** SigLIP needs G-21 error rates on labelled baseline outputs and a §6.2 transfer test
   to block. Options: (a) collect ~150-300 human yes/no labels on the smoke or full baseline images (a person
   viewing the images; ~15-30 min per rater) and run the clause blocking with a registered error bound, the
   transfer test recorded as a prototype gap; (b) run it non-blocking (reported only); (c) drop it. Margin δ = 0.10
   and the top-5 rule are proposals.
4. **Arm directory shape.** One evaluation run writes three arms under `/out/arm-<metric>/`; MM-34 says the rw mount
   is "the arm directory for evaluation". Either `ext-collect` accepts `--eval-* <out>/arm-<metric>` (proposed), or
   the evaluator writes one arm per run (three GPU runs, or a CPU re-score mode for the codec metrics).
5. **MM-38 for the binary constraint metric.** If the 20-item smoke baseline has zero SigLIP violations, the binary
   rule ("degenerate when the mean is 0 or 1") would call it degenerate, although the non-inferiority clause counts
   candidate-only violations and is valid at a zero baseline rate. Apply MM-38 to the deciding metric only?
6. **Epoch count** for the full run is set from the smoke's measured seconds per epoch (default 24 ≈ 1.2 GPU-h at the
   estimate). Learning rate 3e-4 and clip 1e-4 are ddpo-pytorch's; whether 1,536 samples move WebP measurably is
   an estimate; the smoke cut (128 samples) gives a first signal.
7. The ddpo-pytorch clone used for the citations sat inside the read-only mount (never imported); move it out before
   launch to keep the mount minimal.

## Decisions taken (2026-11-12)

The maintainers settled the open questions as follows. 1: port (c) accepted. 2: (a)
`webp_lossless_bpp` is T3, deciding and blocking, construct-aligned (design revision 15, R15-1). 3: (b) constraint
clause reported only, non-blocking. 4: `ext-collect` accepts `<out>/arm-<metric>` (adapter work). 5: MM-38 applies
to the deciding metric only. 6: epochs from the smoke's measured seconds per epoch. 7: the clone moved out of the
runtime directory.
