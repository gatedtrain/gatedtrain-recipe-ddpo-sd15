# Area "image generation, SD1.5" (DDPO + LoRA) gate: evaluation plan (DRAFT)

prototype, unpromotable. Draft written 2026-11-12 (UTC) on the ptc-skel side, before the area's smoke run. The
real files are written under `<data-root>/area3-sd15/gate/` (this draft never touches
the data directory). Every GPU step runs through `ext-launch` and `gpu-step.sh` after approval (MM-31). Adapter
`ddpo_sd15` (tests/README.md section 11.20, EXT-118 to EXT-130). Below, `PTC` is
ptc's command line (`python -m ptc_skel` at the time), `A3` is
`<data-root>/area3-sd15` (the runtime directory), `OUT` is `<data-root>/out/area3-sd15`, `REC` is
`<data-root>/records/area3-sd15`, `REQ` is a scratch directory holding the launch requests.

## 1. Files

| File | Purpose | Hash |
|---|---|---|
| `spec.yaml` | workflow `mm-a3-sd15-ddpo`: stage `rl.ddpo@0.1`, gate `gate.superiority@1.0` on `t2i.sd15_webp_items@0.1` | pins base, prompts, items, trainer config; registration hash and `mde` after the smoke |
| `registration.yaml` | v1; `webp_lossless_bpp`, lower, superiority, margin 0, alpha 0.025, power 0.8, ucb80 on the smoke pilot (s, df 19), planning effect from it, N 150, K 2, seeds 11 and 12, `construct_alignment`; `created_at` to be added | pin in spec.yaml after the smoke values and `created_at` |
| `trainer.yaml` | `{backend: ddpo_sd15, profile: full}` | `sha256:0aff6518...af9ec0` |
| `smoke_pilot.py` | helper: smoke record scores and the sigma pilot from the two filled 20-item arms (stdlib only) | none |

Fixed hashes (computed 2026-11-12 with ptc-skel on the real directories, read only):
- base `A3/models/stable-diffusion-v1-5/` manifest `sha256:1d99f8b61d787ed9d67942b14a61264397ee3ccf66baf5bcd72743b814f56487`
  (13 files; also the baseline arm's `checkpoint.model_hash` and every arm's `base_model_hash`, MM-18);
- training prompts `A3/prompts/` manifest `sha256:a75f8fb3ba33f339a4d63b5d84e9e13638aa71a5d309d8fe31b179d0e13df8ee`
  (the ledger's `data_ref.snapshot`; shard `train.jsonl` `sha256:37dbad02...04e8`, run_args.json `train_file_sha256`);
- item set `A3/items/` manifest `sha256:0a4af3f984aacdd51fe2bed497667f62c50cb487bb991311b854a2c6cd395bd4` (150 items;
  `items.jsonl` plus `item_set_meta.json`; the README's `4a99afa6...` is the file hash of `items.jsonl` only);
- evaluator identity `t2i.sd15_webp_items@0.1` `sha256:b06def21aa01eee0f5dcf3e881a36ebe1f6764a4d2971a06b61c72cf4e71bfb7`
  (scorer code = SHA-256 of `eval_images.py`, `sha256:5575c66f...`; versions pillow 13.0.0, libwebp 1.6.0 as the
  2026-11-12 base evaluation wrote them); supporting `t2i.sd15_jpeg_items@0.1` `sha256:9cb69feb...` (not registered);
- deployment sampler (20 steps, 512 px, DDIM eta 1, CFG 5) RFC 8785 hash
  `sha256:2ff98254d072fd3b4926afe981c90ab46278eb88972497a1479610e78c535ccb` (`eval_images.py` writes the compact-JSON
  hash `sha256:2516f605...`; `ext-arm-fill` converts it).

## 2. Smoke (MM-38, MM-39) and the sigma pilot

1. **Training smoke** (`REQ/smoke.request.yaml`, label `mm-a3-sd15-smoke`, output `OUT/smoke-1`, 2 epochs x 64 samples,
   cuts 1 and 2, gpu_est_h 0.15):
   ```
   mkdir OUT/smoke-1        # new, empty, owned by the launching user
   PTC ext-launch --adapter ddpo_sd15 --profile smoke --request REQ/smoke.request.yaml --gpu-log <host-work>/gpu-log.md --out-record REC/smoke-1.launch.json
   PTC ext-launch --adapter ddpo_sd15 --recheck REC/smoke-1.launch.json
   <the printed gpu-step.sh command>
   ```
   Read from `OUT/smoke-1/memory.jsonl`: `seconds_epoch` (training seconds per epoch), `peak_reserved_gb`; from
   `nvidia-smi` sampling: peak GPU memory.
2. **Smoke evaluations** (`--profile smoke`, same three commands each, outputs `OUT/eval-smoke-base-1` and
   `OUT/eval-smoke-cut-1`): the base alone and smoke cut 2 (mounted read-only at `/inputs/cut`) on the first 20
   registered items, seeds 11 and 12 (`--limit 20`).
3. **Fill the four arms** (deciding and supporting of each run):
   ```
   PTC ext-arm-fill --adapter ddpo_sd15 --checkpoint A3/models/stable-diffusion-v1-5 --base A3/models/stable-diffusion-v1-5 --items A3/items OUT/eval-smoke-base-1/arm-webp_lossless_bpp
   PTC ext-arm-fill --adapter ddpo_sd15 --checkpoint OUT/smoke-1/checkpoint-2 --base A3/models/stable-diffusion-v1-5 --items A3/items OUT/eval-smoke-cut-1/arm-webp_lossless_bpp
   ```
   (and the same for `arm-jpeg_q95_kb`). Expect `filled checkpoint.model_hash`, `filled item_set_hash`, `converted
   sampler_hash`, `filled checkpoint.base_model_hash`. Do not fill `arm-prompt_violation`: it is refused (its extra
   MM-C3 keys; the constraint arm is reported only, section 6).
4. **Pilot and smoke record.** `python3 -I smoke_pilot.py OUT/eval-smoke-base-1/arm-webp_lossless_bpp
   OUT/eval-smoke-cut-1/arm-webp_lossless_bpp` prints the 20 per-item baseline scores, `baseline_pilot_mean`,
   `sigma_pilot.s` (df 19) and `planning_effect`. Write `REC/smoke-1.smoke.json`: `schema ptc_skel.smoke_record`,
   `schema_version 1.0`, `tags`, `adapter ddpo_sd15`, `area 2`, `base_model_hash sha256:1d99f8b6...`, `item_set_hash
   sha256:0a4af3f9...`, `peak_gpu_mem_gb`, `host_ram_peak_gb`, `swap_peak_gb`, `seconds_per_step` (training seconds
   per optimizer step = seconds per epoch / 2), `projected_gpu_h` (full run: <EPOCHS> x seconds per epoch + load and
   save overhead, plus the two 150-item evaluations), `gpu_log_total_h`, `baseline_scores` (the 20 webp scores),
   `error_count` (both smoke arms), `verdict PASS`, `reasons []`. ptc-skel recomputes the verdict on
   `webp_lossless_bpp` only (decision 5): [0, null] has no mean rule; FAIL when 10 or more of the 20 scores share one
   value or their SD is 0.
5. **Registration values.** Replace `<SMOKE_SD_DIFF>`, `<SMOKE_BASELINE_MEAN>`, `<PLANNING_EFFECT>` (also spec.yaml
   `mde`), name the smoke label in `sigma_pilot.source`, add `created_at`, pin the registration hash in spec.yaml. Check:
   `PTC ext-launch ... --spec A3/gate/spec.yaml` loads it (step 3 below). With s = 0.6 the plan is n_required 136 of 150
   (planning effect 0.17, checked with ptc-skel on this draft).
6. **Epochs.** `<EPOCHS>` from the smoke's seconds per epoch and the remaining budget (README default 24, about 1.2
   GPU-h at the estimate of 3 min per epoch). The full profile cuts every 4 epochs and at the last one; choose a
   multiple of 4 so the last cut is a regular one (any value works: the last epoch always saves).

## 3. Full run

`REQ/full.request.yaml` with `<EPOCHS>` set (keep it quoted: `- '24'`; argv items are strings), label `mm-a3-sd15-full`, output `OUT/full-1`:
```
mkdir OUT/full-1
PTC ext-launch --adapter ddpo_sd15 --profile full --request REQ/full.request.yaml --gpu-log <host-work>/gpu-log.md --smoke-record REC/smoke-1.smoke.json --spec A3/gate/spec.yaml --out-record REC/full-1.launch.json
PTC ext-launch --adapter ddpo_sd15 --recheck REC/full-1.launch.json --spec A3/gate/spec.yaml
<the printed gpu-step.sh command>
```
Expect cuts `checkpoint-4`, ..., `checkpoint-<EPOCHS>`, each with `cut.json`; `run_args.json` `seed 3000`, `profile
full`, `settings.epochs <EPOCHS>`, `base_ref.verified true`; `samples.jsonl` 64 lines per epoch; `train_steps.jsonl`
2 steps per epoch.

## 4. Arms

`REQ/eval-base.request.yaml` (base alone, 150 items) and `REQ/eval-cand.request.yaml` (`OUT/full-1/checkpoint-<EPOCHS>`
read-only at `/inputs/cut`), each `--profile full --smoke-record REC/smoke-1.smoke.json` (the GPU estimate becomes the
smoke's projection: conservative; these records are never collected). Then:
```
PTC ext-arm-fill --adapter ddpo_sd15 --checkpoint A3/models/stable-diffusion-v1-5 --base A3/models/stable-diffusion-v1-5 --items A3/items OUT/eval-base-1/arm-webp_lossless_bpp
PTC ext-arm-fill --adapter ddpo_sd15 --checkpoint OUT/full-1/checkpoint-<EPOCHS> --base A3/models/stable-diffusion-v1-5 --items A3/items OUT/eval-cand-1/arm-webp_lossless_bpp
```
(and `arm-jpeg_q95_kb` of both, for the supporting comparison).

## 5. Collect

```
PTC ext-collect A3/gate/spec.yaml --adapter ddpo_sd15 --stage-out OUT/full-1 --launch-record REC/full-1.launch.json --smoke-record REC/smoke-1.smoke.json --eval-baseline OUT/eval-base-1/arm-webp_lossless_bpp --eval-candidate OUT/eval-cand-1/arm-webp_lossless_bpp --runs-root <data-root>/runs/area3-sd15
```
No `PTC_SKEL_EXT_*` override may be set. **Expect:** `metric: webp_lossless_bpp`, `direction: lower_is_better`,
`independence_tier: T3`, a `construct_alignment:` line (what a pass shows), `n_items 150`, `samples_per_item 2`,
unpaired 0/0, `baseline_model_hash sha256:1d99f8b6...`, `candidate_model_hash` = the composed hash `ext-arm-fill`
printed, `contamination: exact_overlap=0` (items against the 248 shard prompts and every sampled prompt),
`consumption_precision: row` (observed; one ledger record per sample, 64 x <EPOCHS>), `gpu_seconds` = the wrapper time.
`PTC impact A3/prompts/train.jsonl --run-dir <run>` returns every cut with answer `consumed` at `row`.

The supporting T0 comparison (`jpeg_q95_kb`, the reward) is read from the two filled `arm-jpeg_q95_kb` directories; a
second collect gating on `t2i.sd15_jpeg_items@0.1` would need its own registration and prints "non-blocking under 05
section 6.1; prototype plumbing only".

## 6. The constraint (reported only)

`arm-prompt_violation` (SigLIP top-5, 1 = the prompted animal is not depicted) is written by every evaluation run but
not collected: ext-collect refuses constraint clauses (REG-CONSTRAINT) and the evaluator has no registry row
(decision 3; wiring is a later task). Report it beside the gate from the two 150-item arms: per item v_i = max over the
two repeats; baseline and candidate violation rates; c (candidate-only) and b (baseline-only) counts; the exact
Clopper-Pearson upper bound of c/n at alpha 0.025 against the README's proposed delta 0.10; label it "non-blocking,
no human-labelled error rates (G-21) and no transfer test (05 section 6.2)".

## 7. Assumption: the evaluated base is the default one

`eval_images.py` loads `--base` (default `A3/models/stable-diffusion-v1-5`, inside the read-only identity mount) without
hashing it, and its outputs do not record which base it loaded; `ext-arm-fill --base` then states the base, it does not
observe it (review 7, L1). Every evaluation request therefore uses the default base: none of the four eval requests
passes `--base`. `ext-launch` and `--recheck` enforce it for `ddpo_sd15` (EXT-133): an `eval_images.py` argv with a
`--base` other than `<script dir>/models/stable-diffusion-v1-5` is refused. A runtime-side binding (the evaluator
recording the resolved base and its `files_sha256`, checked by fill) is a request to the area owner, not done.

## 8. Open items

1. The smoke record's `baseline_scores` are per-item means over the two repeats (20 values). The MM-38 rules were
   written for K = 1 areas; per (item, repeat) scores (40 values) would also work. Choice: per-item means.
2. The sigma pilot pairs the smoke cut (epoch 2, 128 samples) with the base; the full run's cut is further from the base,
   so sigma may grow: ucb80 adds 18 % at df 19.
3. `PYTORCH_CUDA_ALLOC_CONF` is allow-listed but not in the requests; add `expandable_segments:True` only if the smoke
   shows fragmentation.
