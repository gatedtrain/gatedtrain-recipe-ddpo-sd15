Prototype result of a single registered run; not a benchmark claim; not affiliated with or endorsed by the owners of the models, datasets, or frameworks used.

# Result (2026-11-13): gate PASS

DDPO port on diffusers 0.42.1 and peft 0.21.2 (LoRA rank 4) on Stable Diffusion 1.5, JPEG-compressibility reward,
12 epochs of 64 samples (768 samples; cuts 4, 8 and 12; 0.41 GPU-hours on one RTX 4090). The epoch count came from
the smoke run (2 epochs already moved the pilot by -0.11 bpp, about twice the planning effect). Candidate: cut 12;
baseline: the plain base.

| | |
|---|---|
| Deciding metric | `webp_lossless_bpp`, lower is better (T3: libwebp lossless, another engine than the reward's libjpeg; construct-aligned with the reward, so a pass shows that the reward's gain transfers to another codec, not that the images are better) |
| Baseline (SD1.5) | 9.811 bpp |
| Candidate (cut 12) | 9.382 bpp |
| Improvement, 95% cluster-bootstrap interval | 0.4288 bpp [0.3748, 0.4859], p <= 6.2e-05 |
| Items, samples per item | 150, 2 (seeds 11 and 12) |
| Achieved power | 0.567 at the declared MDE 0.06 (reported `underpowered`: the observed sd of paired differences, 0.345, exceeds the smoke pilot's ucb80 sigma, 0.242, because the full cut moves further from the base than the smoke cut; the interval is far from 0, so the pass stands and the pilot's underestimate is disclosed) |
| Exact overlap with training | 0 |
| Consumption precision | `row` (observed, 768 records) |

Supporting (T0, the reward itself): JPEG q95 104.9 kB to 96.0 kB. Constraint (SigLIP top-5 "the prompted animal is
depicted", reported only, non-blocking): violation rate 9.3 % in both arms, 2 candidate-only and 2 baseline-only
violations of 150; the Clopper-Pearson upper bound of 2/150 at 0.025 is about 0.047, below the proposed delta 0.10.
The impact query on `prompts/train.jsonl` returns cuts 4, 8 and 12 (steps 8, 16 and 24) as consumed at `row`.

Re-collected on 2026-11-16 with this repository's adapter (installed as a plugin of the framework, then named ptc;
ptc, ptc-skel and ptc-hub are earlier names of gatedtrain, see "Rename to gatedtrain 0.3.0") on
the original run's outputs: every field of the gate report equals the re-collect with the framework before the move, except the run-id-derived
`lockfile_hash` (the lockfile differs in its run id, the ledger artifact hash derived from it, and the
`checkpoint_completeness` sources, which now name `runtime/` files); the 768 ledger rows and the lineage are equal,
and the impact query returns the same three cuts.

## Published files

The files under results/2026-11-13/ are redacted copies (host paths replaced by `<data-root>` and `<host-work>`,
internal process references removed) of `gate/EVAL-PLAN.md`, `gate/registration.yaml` (`gate/spec.yaml` pins the
original's hash) and `gate/spec.yaml`; the gate-time originals are not published. `gate/trainer.yaml` is unchanged
(`0aff6518257f2c7371d67874bc4b5f5a39e5c2a7eb36220cf2e2615778af9ec0`).

The published runtime scripts differ from the ones that produced this result only by a comment and docstring
cleanup before publication (the original scripts are not published): `runtime/eval_images.py` (now
`sha256:77ad333fe9a9d8b93a487a1a00f3a6995c098a091c0955539b1cf3a19fb28259`) and `runtime/train_ddpo.py` (now
`sha256:08b5d23b974e1da414a03f036d33a034b5a214a805db979c75fdc0ba45ef52b0`); their shared sampler block, and so
`sampler_code_sha256`, is unchanged. The scorer pin `code_sha256` in `rows.py` followed the published file, so the
identity hash of `t2i.sd15_webp_items@0.1` became
`sha256:1aab8700b323eea7573d2b321dcc871b5cd995dc73a3185b2a02cb96cede7f99` instead of the
`sha256:b06def21aa01eee0f5dcf3e881a36ebe1f6764a4d2971a06b61c72cf4e71bfb7` that the gate registration pins (it changed
again with the rename below). The result above was re-collected unchanged on 2026-11-16, before this cleanup.

## Rename to ptc-hub 0.2.0

ptc-hub 0.2.0 renamed the persisted identifiers (schema ids, ledger writer id, analysis code version). The files under
results/2026-11-13/ keep the names ptc-skel 0.1.0 wrote, as do the run's records and outputs; they are read by ptc
`78eb913` (this and the other framework commits cited here, `e20736e` and `372c315` among them, are unpublished
pre-release commits of the framework, earlier named ptc-skel and ptc-hub; the published framework is `gatedtrain`
`v0.3.0`), and the gate verdict PASS above remains reproducible with ptc `78eb913`, the original
`gate/registration.yaml` and `gate/spec.yaml`, and the original runtime scripts (none of them published). The gate definition is carried forward by results/2026-11-13/gate/registration-ptc-hub.yaml (amends
`sha256:9124880d06181ef3275dac2e61f29cafa77c69150b8766f822bd1d6ce66f129c`, the published `gate/registration.yaml`, which
differs from the gate-time file only in redacted comments and in the reference inside the free-text
`sigma_pilot.source`; post_hoc: true) and spec-ptc-hub.yaml. Re-pinned for the rename:
`runtime/eval_images.py` `sha256:77ad333fe9a9d8b93a487a1a00f3a6995c098a091c0955539b1cf3a19fb28259` ->
`sha256:20e5bc2466a5b13f2d6942d495e7b2d9a99793526fc48aef6b939b7631ea866e` (it writes the schema id
`ptc_hub.item_eval`; its docstring names ptc-hub; historical runs used the older versions; only the new pin is
accepted).

The amendment is post hoc (05 section 3.7): the scorer the registration pins changed after the gate by the
publication cleanup, before the rename, so the amendment is not version-only and also carries the new evaluator
identity `sha256:5147a146397a867ae7acc01b11a97a0c80e9150319f34fb1477ae5a82bea3812` (an evaluator change, not only a
rename; restoring the gate-time scorer would undo the cleanup). Under ptc-hub 0.2.0 the historical PASS is therefore
reported INCONCLUSIVE with reason `post_hoc`; the original verdict reproduces at ptc `78eb913` as described above.

Re-judged on 2026-11-17 with ptc `e20736e` on a scratch replay of the run (nothing under the run's directories was
written): copies of `full-1.launch.json`, `smoke-1.smoke.json` and the two `arm-webp_lossless_bpp` arms with only the
schema ids replaced by the ptc_hub ids and, in the arms' `eval_meta.json`, `scorer.code_sha256` rewritten from the
gate-time pin to the new pin `sha256:20e5bc24...` (scratch copies only); in the launch record copy
the values derived from its content were recomputed with ptc's own functions on a scratch host-work directory (the
hashed `label` with `launch_label`, and `host_work`, `gpu_step_sha256`, `host_config_sha256`; the guard-log lines of the
run copied under the new label). `ptc-hub ext-collect results/2026-11-13/gate/spec-ptc-hub.yaml --adapter ddpo_sd15`
(`<data-root>` substituted, runtime override at the original runtime directory, the original input roots) gave status
INCONCLUSIVE, reasons `[post_hoc]`, `post_hoc: true`, `registration_amends sha256:9124880d...` (the published copy). Against the published
report every other field is equal, among them `delta_hat` 0.4288, the interval [0.3748, 0.4859], `p_value`,
`achieved_power` 0.567, `mde`, `sd_diff`, `n_items` 150, the baseline and candidate scores, the item-set, sampler,
model and checkpoint hashes, contamination, consumption precision and `gpu_seconds`, except: `status`/`status_reasons`
(PASS -> INCONCLUSIVE `post_hoc`), `post_hoc`, `registration_hash`, `registration_amends`, `evaluator_hash` and
`evaluator_hash_registered` (`sha256:5147a146...`), `analysis_code_version`, `schema`, `lockfile_hash`, `run_id` and
`created_at`. The 768 ledger rows are equal except `writer_id` (`ptc-hub.ledger@0.2.0`) and the run-id-derived
`run_id` and `sample_id`; the lineage is equal except the registration hash and run-id-derived activity ids.

## Rename to gatedtrain 0.3.0 (2026-11-18)

The framework was renamed again, from ptc-hub to gatedtrain (distribution and import `gatedtrain`, version 0.3.0),
and this recipe with it: distribution `gatedtrain-recipe-ddpo-sd15`, import package `gatedtrain_recipe_ddpo_sd15`,
entry-point group `gatedtrain.adapters`, runtime override `GATEDTRAIN_RECIPE_DDPO_SD15_RUNTIME`, runtime project
`gatedtrain-recipe-ddpo-sd15-runtime`. gatedtrain 0.3.0 renamed the persisted identifiers once more (schema ids
`gatedtrain.*`, ledger writer id `gatedtrain.ledger@0.3.0`, analysis code version `gatedtrain 0.3.0`). The files
under results/2026-11-13/ keep the bytes they had: `registration.yaml`, `spec.yaml`, `EVAL-PLAN.md` and
`trainer.yaml` the names ptc-skel 0.1.0 wrote, `registration-ptc-hub.yaml` and `spec-ptc-hub.yaml` the ptc-hub
0.2.0 names, as do the run's records and outputs. The recipe-runtime schema ids `ptc_mm.*` (in `runtime/` data files,
the scripts that write them, and the run's cuts, models and item set, whose manifest hashes the registration pins)
are data-format identifiers recorded by the published run and are kept.

Changed beyond the name map: `runtime/eval_images.py` now carries in its docstring the MIT attribution of the code it
ports from kvablack/ddpo-pytorch (as `runtime/train_ddpo.py` does), and `THIRD_PARTY_NOTICES.md` names the
huggingface/diffusers files that ddpo-pytorch's copies come from. Re-pinned: `runtime/eval_images.py`
`sha256:20e5bc2466a5b13f2d6942d495e7b2d9a99793526fc48aef6b939b7631ea866e` ->
`sha256:504b374e820f5684e5f5e1c4f9bb0584e18edc9a36d8b12996fcca7e709331c7` (the attribution, the schema id
`gatedtrain.item_eval` and the framework name in its docstring; its shared sampler block, and so
`sampler_code_sha256`, is unchanged; only the new pin is accepted). `runtime/uv.lock`
`sha256:cdd914fae6bd487e20d5ec8bfac8eb15ffca0d32fb326edae935689183102f3b` ->
`sha256:850fc4c129bd2d71543f7c28660f1b20fbd89ef9b90a83cd2e72f113bb1c3077` (regenerated for the runtime project's new
name and license field; the 67 locked packages and their versions are unchanged). `runtime/train_ddpo.py`,
`runtime/fetch_models.py`, the items, prompts, constraint set and model pins are unchanged.

The gate definition is carried forward by results/2026-11-13/gate/registration-gatedtrain.yaml (amends
`sha256:c6fdb8459d226c6063c1071dca5028839b18f7ded43f562014bbcb791f42ddbf`, `registration-ptc-hub.yaml`; post_hoc:
true) and spec-gatedtrain.yaml. Besides `analysis_code_version`, `amends` and `created_at`, the only field that
differs from the prior amendment is the evaluator identity of `t2i.sd15_webp_items@0.1`,
`sha256:5147a146...` -> `sha256:fed58f5a80d20553688c13fe8dff50e4331fc50823ab7f9c69bc6e2c5d02c270` (the scorer
changed beyond the rename, so this is not a version-only amendment; it stays post hoc, as the amendment it carries
forward). The original verdict PASS remains reproducible with ptc `78eb913` as described above.

Re-judged on 2026-11-18 with gatedtrain `372c315` on a scratch replay, as in the previous section (nothing under the
run's directories was written): copies of `full-1.launch.json`, `smoke-1.smoke.json` and the two
`arm-webp_lossless_bpp` arms with only the schema ids replaced by the gatedtrain ids and, in the arms'
`eval_meta.json`, `scorer.code_sha256` rewritten from the gate-time pin to the new pin
`sha256:504b374e...`; in the launch record copy the hashed `label`, `host_work`, `gpu_step_sha256` and
`host_config_sha256` recomputed with gatedtrain's own functions on a scratch host-work directory (the run's two
guard-log lines copied under the new label). `gatedtrain validate results/2026-11-13/gate/spec-gatedtrain.yaml`
(`<data-root>` substituted) exits 0; `spec-ptc-hub.yaml` is refused (REG-FIELD, old analysis code version, with the
amendment fix). `gatedtrain ext-collect results/2026-11-13/gate/spec-gatedtrain.yaml --adapter ddpo_sd15` (runtime
override at the original runtime directory, the original input roots and stage output) gave status INCONCLUSIVE,
reasons `[post_hoc]`, `post_hoc: true`, `registration_amends sha256:c6fdb845...`. Against the published report every
other field is equal, among them `delta_hat` 0.4288, the interval [0.3748, 0.4859], `p_value` 6.2e-05,
`achieved_power` 0.567, `n_items` 150, the baseline (9.8112) and candidate (9.3824) scores, the item-set and sampler
hashes, except: `status`/`status_reasons` (PASS -> INCONCLUSIVE `post_hoc`), `post_hoc`, `registration_hash`
(`sha256:40174624...`), `registration_amends`, `evaluator_hash` and `evaluator_hash_registered`
(`sha256:fed58f5a...`), `analysis_code_version`, `schema`, `lockfile_hash`, `run_id` and `created_at`. Against the
2026-11-17 ptc-hub re-judgement only `analysis_code_version`, `schema`, the registration hashes, the evaluator
hashes, `lockfile_hash`, `run_id` and `created_at` differ. The 768 ledger rows are equal except `writer_id`
(`gatedtrain.ledger@0.3.0`) and the run-id-derived `run_id` and `sample_id`; the lineage is equal except the
registration hash and run-id-derived ids.
