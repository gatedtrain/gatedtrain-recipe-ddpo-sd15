"""External-run adapters, slice 3: area "image generation, SD1.5" (DDPO + LoRA; tests/README.md section 11.20,
EXT-118 to EXT-130).

Sources: MULTIMODAL-SPEC.md MM-1 to MM-39 (section 6 row 2a and its accepted deviations, MM-17, MM-18, MM-21, MM-24,
MM-27, MM-38), design revision 15: 05 section 6.1 (tiers; "Same programmatic tool", R15-1: a construct-aligned T3
metric, stated in the manifest, the registration and the report), section 11 (lineage, the `lora_compose` activity of
MM-18), section 12.1 and 12.2 (a `row` record: data_ref.row, seed inputs, sample id), section 11.2 (derivation
`observed`); runtime/README.md "Decisions taken"; the runtime `runtime/`
(README "What gatedtrain needs", `train_ddpo.py`, `eval_images.py`, read 2026-11-12). Helpers: `tests/_ddpo.py`.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from _ddpo import (ADAPTER, BASE_FILES, BASE_NAME, CUT_FILES, DDPO, DEPLOYABLE, ENV_NAMES, EVAL_IMAGES,
                   EVAL_IMAGES_SHA256, EVALUATOR_ID, LORA_WEIGHTS, METRIC, REPO, REVISION, RUNTIME_CONTAINER, SAMPLER,
                   SAMPLER_KEYS, SCORER, SEEDS, SUPPORT_ID, SUPPORT_METRIC, SUPPORT_SCORER, TRAINER_ID, DdpoScenario,
                   adapter_hash, baseline_score, build_base, build_cut, compact_hash, lora_model_hash, read_jsonl,
                   run_user, smoke_record, write_json, write_jsonl)
from gatedtrain.testing.ext import IMAGE_ID, sha_of, write_canonical
from gatedtrain.testing.support import EXIT_OK, EXIT_REFUSED, describe, file_sha256, manifest_hash, parse_errors, parse_run_dir, run_cli


def registry():
    from gatedtrain.ext import registry as reg
    return reg


def adapter():
    return registry().adapter(ADAPTER)


class DdpoCase(unittest.TestCase):
    """One scenario per class; each test that changes it builds its own (`fresh`)."""

    N_ITEMS = 12

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-ddpo-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.n = 0

    def fresh(self, **kw) -> DdpoScenario:
        type(self).n += 1
        kw.setdefault("n_items", self.N_ITEMS)
        return DdpoScenario(self.tmp / f"sc-{type(self).n}", **kw)

    def runs(self) -> Path:
        type(self).n += 1
        return self.tmp / f"runs-{type(self).n}"

    def passes(self, sc: DdpoScenario, **kw) -> Path:
        proc = sc.collect(self.runs(), **kw)
        self.assertIn(proc.returncode, (EXIT_OK, 1), describe(proc))
        return parse_run_dir(proc.stdout)

    def refused(self, rule: str, field_prefix: str | None = None, *, sc: DdpoScenario, text: str | None = None, **kw):
        runs = self.runs()
        proc = sc.collect(runs, **kw)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        hits = [e for e in errs if e["rule"] == rule and (field_prefix is None or e["field"].startswith(field_prefix))
                and (text is None or text in e["message"])]
        self.assertTrue(hits, f"expected error[{rule}] {field_prefix}... {text!r}; got {errs}\n{describe(proc)}")
        self.assertFalse(runs.exists() and any(runs.iterdir()), "a refusal writes no run directory")
        return hits


# ---------------------------------------------------------------------------
# EXT-118: registry rows
# ---------------------------------------------------------------------------

class TestRegistryRows(unittest.TestCase):

    def test_trainer_row(self):
        row = registry().TRAINER_ROWS[ADAPTER]
        self.assertEqual(list(row["uses"]), [TRAINER_ID])
        self.assertEqual(list(row["evaluators"]), [EVALUATOR_ID, SUPPORT_ID])        # deciding first (MM-38)
        self.assertEqual(row["consumption"], {"target": "row", "fallback": None})
        self.assertEqual(set(row["pins"]), {"diffusers", "peft", "torch", "transformers"})
        self.assertEqual(row["framework"], "diffusers")
        self.assertEqual(tuple(row["components"]), DEPLOYABLE)
        self.assertFalse(row["spec_binding"])

    def test_evaluator_rows(self):
        rows = registry().EVALUATOR_ROWS
        for eid, metric, tier, scorer in ((EVALUATOR_ID, METRIC, "T3", SCORER),
                                          (SUPPORT_ID, SUPPORT_METRIC, "T0", SUPPORT_SCORER)):
            with self.subTest(evaluator=eid):
                row = rows[eid]
                self.assertEqual((row["metric"], row["direction"], tuple(row["score_range"]), row["tier"]),
                                 (metric, "lower_is_better", (0, None), tier))
                self.assertFalse(row["binary"])
                self.assertEqual(tuple(row["error_classes"]), ("scorer_error",))
                self.assertEqual(row["payload_text_field"], "prompt")
                self.assertEqual(tuple(row["item_payload"]), ("prompt", "label"))
                self.assertFalse(row["normalizer"] or row["env_version"])
                self.assertEqual(set(row["sampler_keys"]), set(SAMPLER_KEYS))
                self.assertEqual(row["scorer"], scorer)
                self.assertEqual(row["samples"], "150, 2")

    def test_construct_alignment_declared(self):
        """R15-1: the WebP metric is T3 and construct-aligned with the JPEG reward; its manifest (the registry row)
        says so; the JPEG arm is the reward itself (T0) and declares nothing."""
        rows = registry().EVALUATOR_ROWS
        ca = rows[EVALUATOR_ID]["construct_aligned"]
        self.assertEqual(ca["reward"], "jpeg_compressibility")
        self.assertIn("another codec", ca["statement"])
        self.assertIn("not that", ca["statement"])
        self.assertIsNone(rows[SUPPORT_ID].get("construct_aligned"))

    def test_identity_hash(self):
        self.assertEqual(registry().evaluator_identity_hash(EVALUATOR_ID),
                         sha_of({"evaluator": EVALUATOR_ID, "metric": METRIC, "direction": "lower_is_better",
                                 "scorer": SCORER}))

    @unittest.skipUnless(EVAL_IMAGES.is_file(), "the area-3 runtime is absent")
    def test_scorer_code_hash_is_eval_images(self):
        self.assertEqual(file_sha256(EVAL_IMAGES), EVAL_IMAGES_SHA256)

    def test_constraint_evaluator_not_registered(self):
        """The SigLIP constraint (prompt_violation) is reported only, outside ext-collect, until its constraint-clause
        wiring (decision 3, 2026-11-12): no registry row, so no gate can name it."""
        self.assertNotIn("t2i.sd15_animal_presence@0.1", registry().EVALUATOR_ROWS)

    def test_item_set_rows(self):
        """The real items: `prompt` and `label` strings (class_index, an integer, is not a payload field)."""
        from gatedtrain.ext.collect import read_item_set
        row = registry().EVALUATOR_ROWS[EVALUATOR_ID]
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ddpo-items-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copyfile(DDPO / "items.jsonl", tmp / "items.jsonl")
        rows, bad = read_item_set(tmp, tuple(row["item_payload"]), tuple(row.get("item_payload_objects", ())))
        self.assertEqual((len(rows), bad), (150, []))


# ---------------------------------------------------------------------------
# EXT-119: cuts and completeness
# ---------------------------------------------------------------------------

class TestCuts(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ddpo-cuts-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.base = build_base(self.tmp / "base")
        self.out = self.tmp / "run"
        for e, gs in ((2, 4), (4, 8), (6, 12)):
            build_cut(self.out, e, global_step=gs, samples_consumed=gs * 4, base=self.base, script_sha256="sha256:" + "1" * 64)

    def found(self):
        complete, incomplete = adapter().find_cuts(self.out)
        return [s for s, _ in complete], sorted(Path(p).name for p in incomplete)

    def test_complete_by_global_step(self):
        """find_cuts gives (last_step_included, path): the cut's optimizer step (cut.json global_step), not the epoch
        in the directory name."""
        self.assertEqual(self.found(), ([4, 8, 12], []))
        self.assertEqual([e for e, _ in adapter().checkpoint_dirs(self.out)], [2, 4, 6])

    def test_partial_directory_ignored(self):
        shutil.copytree(self.out / "checkpoint-6", self.out / ".checkpoint-8.partial")
        self.assertEqual(self.found(), ([4, 8, 12], []))

    def test_incomplete_cuts(self):
        def edit(d, fn):
            doc = json.loads((d / "cut.json").read_text()); fn(doc)
            (d / "cut.json").write_text(json.dumps(doc))
        def no_marker(d): (d / "cut.json").unlink()
        def changed(d): (d / "adapter_config.json").write_text("{}")
        def wrong_epoch(d): edit(d, lambda doc: doc.update(epoch=5))
        def extra(d): (d / "notes.txt").write_text("x")
        def missing(d): (d / "trainer_state.json").unlink()
        def unlisted(d): edit(d, lambda doc: doc["files"].pop("sampler.json"))
        def deployable(d): edit(d, lambda doc: doc.update(deployable=[LORA_WEIGHTS, "adapter_config.json"]))
        def no_global_step(d): edit(d, lambda doc: doc.pop("global_step"))
        def truncated(d):
            data = (d / LORA_WEIGHTS).read_bytes()[:-4]
            (d / LORA_WEIGHTS).write_bytes(data)
            edit(d, lambda doc: doc["files"].update({LORA_WEIGHTS: file_sha256(d / LORA_WEIGHTS)}))
        def marker_garbage(d): (d / "cut.json").write_text("{not json")
        for fn in (no_marker, changed, wrong_epoch, extra, missing, unlisted, deployable, no_global_step, truncated,
                   marker_garbage):
            with self.subTest(case=fn.__name__):
                shutil.rmtree(self.out / "checkpoint-4")
                build_cut(self.out, 4, global_step=8, samples_consumed=32, base=self.base,
                          script_sha256="sha256:" + "1" * 64)
                fn(self.out / "checkpoint-4")
                self.assertEqual(self.found(), ([4, 12], ["checkpoint-4"]))

    def test_base_directory_is_a_model_directory(self):
        """The plain SD1.5 base (13 files, whole safetensors) is a model directory; a truncated weight is not."""
        self.assertIsNone(adapter().model_dir_problem(self.base))
        self.assertIsNone(adapter().model_dir_problem(self.out / "checkpoint-6"))
        w = self.base / "unet" / "diffusion_pytorch_model.fp16.safetensors"
        w.write_bytes(w.read_bytes()[:-3])
        self.assertIsNotNone(adapter().model_dir_problem(self.base))
        (self.tmp / "empty").mkdir()
        self.assertIsNotNone(adapter().model_dir_problem(self.tmp / "empty"))


# ---------------------------------------------------------------------------
# EXT-120: identity (MM-17, MM-18)
# ---------------------------------------------------------------------------

class TestIdentity(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ddpo-id-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.base = build_base(self.tmp / "base")
        self.cut = build_cut(self.tmp / "run", 2, global_step=4, samples_consumed=16, base=self.base,
                             script_sha256="sha256:" + "1" * 64)

    def test_base_alone(self):
        """MM-18: the baseline arm evaluates the base alone; its model hash is the base's manifest hash."""
        ident = adapter().identity(self.base)
        self.assertEqual(ident["model_hash"], manifest_hash(self.base))
        self.assertEqual(ident["cut_hash"], manifest_hash(self.base))
        self.assertIsNone(ident["base_ref"])

    def test_lora_cut(self):
        ident = adapter().identity(self.cut, self.base)
        self.assertEqual(ident["model_hash"], lora_model_hash(self.cut, self.base))
        self.assertEqual(ident["adapter_hash"], adapter_hash(self.cut))
        self.assertEqual(ident["cut_hash"], manifest_hash(self.cut))
        self.assertEqual(sorted(ident["components"]), sorted(DEPLOYABLE))
        self.assertEqual(ident["base_ref"], {"repo": REPO, "revision": REVISION,
                                             "manifest_hash": manifest_hash(self.base)})

    def test_resume_state_not_a_component(self):
        before = adapter().identity(self.cut, self.base)["model_hash"]
        (self.cut / "trainer_state.json").write_text('{"epoch": 2, "other": 1}\n')
        self.assertEqual(adapter().identity(self.cut, self.base)["model_hash"], before)
        (self.cut / "sampler.json").write_text('{"num_steps": 50}\n')        # the deployment sampler is a component
        self.assertNotEqual(adapter().identity(self.cut, self.base)["model_hash"], before)

    def test_base_changes_the_model_hash(self):
        before = adapter().identity(self.cut, self.base)["model_hash"]
        other = build_base(self.tmp / "other-base", tag="other")
        self.assertNotEqual(adapter().identity(self.cut, other)["model_hash"], before)

    def test_cut_without_base_refused(self):
        """A LoRA cut's model hash composes the base (MM-18): without the base directory there is no model hash."""
        with self.assertRaises(ValueError):
            adapter().identity(self.cut)


# ---------------------------------------------------------------------------
# EXT-128: the whole run collected (and EXT-120, EXT-123, EXT-130 outputs)
# ---------------------------------------------------------------------------

class TestCollectPass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-ddpo-pass-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = DdpoScenario(cls.tmp / "sc")
        cls.proc = cls.sc.collect(cls.tmp / "runs")
        cls.run_dir = parse_run_dir(cls.proc.stdout) if cls.proc.returncode == EXIT_OK else None

    def setUp(self):
        self.assertEqual(self.proc.returncode, EXIT_OK, describe(self.proc))

    def report(self) -> dict:
        return json.loads((self.run_dir / "report" / "gate_report.json").read_text(encoding="utf-8"))

    def test_pass_and_hashes(self):
        rep = self.report()
        self.assertEqual(rep["status"], "PASS")
        self.assertEqual((rep["adapter"], rep["independence_tier"], rep["direction"]),
                         (ADAPTER, "T3", "lower_is_better"))
        self.assertEqual(rep["baseline_model_hash"], self.sc.base_model)
        self.assertEqual(rep["candidate_model_hash"], self.sc.cand_model)
        self.assertEqual(rep["contamination"], {"exact_overlap": 0, "allowed": False})
        self.assertEqual(rep["consumption_precision"], "row")
        self.assertEqual((rep["n_items"], rep["samples_per_item"]), (150, 2))
        self.assertGreater(rep["delta_hat"], 0)                          # oriented: lower bpp is better (MM-8)

    def test_construct_alignment_reported(self):
        """R15-1: the report states what a pass shows, from the evaluator manifest; the text report too."""
        rep = self.report()
        ca = rep["construct_alignment"]
        self.assertEqual(ca["reward"], "jpeg_compressibility")
        self.assertIn("another codec", ca["statement"])
        text = (self.run_dir / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertIn("construct_alignment: ", text)
        self.assertIn("not that the images are better", text)

    def test_gpu_accounting(self):
        rep = self.report()
        self.assertEqual((rep["gpu_seconds"], rep["wall_seconds"]), (3600, 3600))

    def test_ledger_rows(self):
        """EXT-123 (MM-24): one observed `row` record per generated sample of epochs 1-3; data_ref.row =
        sha256(prompt_id) in shard train.jsonl of the prompt snapshot; step = the optimizer step whose sample_ids
        hold the sample; seed inputs = the noise seed; sample_id = H(run, attempt, epoch, step, prompt_id, seed,
        member)."""
        import hashlib
        import pyarrow.parquet as pq
        rows = pq.read_table(self.run_dir / "ledger" / "consumption.parquet").to_pylist()
        samples = read_jsonl(self.sc.stage_out / "samples.jsonl")
        steps = read_jsonl(self.sc.stage_out / "train_steps.jsonl")
        step_of = {sid: st["step"] for st in steps for sid in st["sample_ids"]}
        self.assertEqual(len(rows), len(samples))
        snapshot = manifest_hash(self.sc.data)
        for r, s in zip(rows, samples):
            self.assertEqual((r["precision"], r["derivation"], r["outcome"]), ("row", "observed", "consumed"))
            self.assertEqual(r["data_ref"]["snapshot"], snapshot)
            self.assertEqual(r["data_ref"]["shard"], "train.jsonl")
            self.assertEqual(r["data_ref"]["row"], "sha256:" + hashlib.sha256(s["prompt_id"].encode()).hexdigest())
            self.assertEqual(r["step"], step_of[s["sample_id"]])
            self.assertEqual(r["seed_inputs"], {"seed": s["seed"]})
            self.assertEqual(r["rewards"], [s["reward"]])
            self.assertIsNone(r["chunk"])
            # NUL-joined text, as the other adapters' ids: the 63-bit noise seed exceeds RFC 8785's safe integers.
            key = "\x00".join(str(x) for x in (r["run_id"], r["attempt_id"], s["epoch"], r["step"], s["prompt_id"],
                                               s["seed"], s["member"]))
            self.assertEqual(r["sample_id"], "sha256:" + hashlib.sha256(key.encode()).hexdigest())
        self.assertEqual(len({r["sample_id"] for r in rows}), len(rows))

    def test_lineage(self):
        """EXT-120 (MM-18): the training stage `used` the base (manifest hash = model hash); each cut is generated
        with its composed model hash at its global step; one `lora_compose` activity per cut used the base and the
        adapter."""
        edges = json.loads((self.run_dir / "lineage" / "edges.json").read_text(encoding="utf-8"))
        train = [a for a in edges["activities"] if a["kind"] == "training_stage"][0]
        model = [u for u in train["used"] if u["role"] == "model"][0]
        self.assertEqual((model["hash"], model["model_hash"]), (self.sc.base_model, self.sc.base_model))
        self.assertEqual([g["last_step_included"] for g in train["generated"]], [4, 6])
        self.assertEqual([g["model_hash"] for g in train["generated"]],
                         [lora_model_hash(c, self.sc.base) for c in self.sc.cuts])
        compose = [a for a in edges["activities"] if a["kind"] == "lora_compose"]
        self.assertEqual(len(compose), 2)
        for act, cut in zip(compose, self.sc.cuts):
            used = {u["role"]: u["hash"] for u in act["used"]}
            self.assertEqual(used, {"base": self.sc.base_model, "adapter": adapter_hash(cut)})
            self.assertEqual(act["generated"][0]["model_hash"], lora_model_hash(cut, self.sc.base))
        gate = [a for a in edges["activities"] if a["kind"] == "gate"][0]
        self.assertEqual([u for u in gate["used"] if u["role"] == "baseline"][0]["hash"], self.sc.base_model)

    def test_impact_on_the_shard(self):
        """A `row` path answers "consumed" (05 section 11.2); both cuts are returned from step 1."""
        proc = run_cli(["impact", str(self.sc.data / "train.jsonl"), "--run-dir", str(self.run_dir), "--json"],
                       cwd=self.sc.root)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        rows = json.loads(proc.stdout)
        rows = rows["results"] if isinstance(rows, dict) else rows
        self.assertEqual([(r["last_step_included"], r["first_step"], r["precision"], r["answer"]) for r in rows],
                         [(4, 1, "row", "consumed"), (6, 1, "row", "consumed")])

    def test_arm_images_copied(self):
        """The arm-<metric> shape: output_ref images (hard links inside the arm) are copied into the run."""
        line = read_jsonl(self.run_dir / "eval" / "candidate" / "items.jsonl")[0]
        self.assertEqual(file_sha256(self.run_dir / "eval" / "candidate" / line["output_ref"]["path"]),
                         line["output_ref"]["sha256"])


# ---------------------------------------------------------------------------
# EXT-121: base binding (MM-18, MM-21)
# ---------------------------------------------------------------------------

class TestBaseBinding(DdpoCase):

    def test_base_ref_files_differ(self):
        """Self-consistent but another base: files, files_sha256 and run_args.json agree with each other."""
        sc = self.fresh()
        def other(d):
            d["files"]["model_index.json"] = "sha256:" + "0" * 64
            d["files_sha256"] = compact_hash(d["files"])
        for cut in sc.cuts:
            sc.edit_json(cut / "base_ref.json", other)
            sc.reseal(cut)
        fs = json.loads((sc.final_cut / "base_ref.json").read_text())["files_sha256"]
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["base_ref"].update(files_sha256=fs))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="not the SHA-256 of the spec's base")

    def test_revision_not_a_commit(self):
        """A branch name, consistent in base_ref.json, adapter_config.json and run_args.json."""
        sc = self.fresh()
        for cut in sc.cuts:
            sc.edit_json(cut / "base_ref.json", lambda d: d.update(revision="main"))
            sc.edit_json(cut / "adapter_config.json", lambda d: d.update(base_model_revision="main"))
            sc.reseal(cut)
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["base_ref"].update(revision="main"))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="40-hex")

    def test_base_model_not_the_spec_base(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(base_model="/inputs/a3/models/other"))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="base_model")

    def test_base_not_verified(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["base_ref"].update(verified=False))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="verified")

    def test_adapter_config_names_another_base(self):
        sc = self.fresh()
        sc.edit_json(sc.final_cut / "adapter_config.json",
                     lambda d: d.update(base_model_name_or_path="runwayml/stable-diffusion-v1-5"))
        sc.reseal(sc.final_cut)
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="adapter_config.json")

    def test_base_mount_missing(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(base_model="/models/sd15"))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc)

    def test_cut_as_the_spec_base_refused(self):
        """The spec's base is the plain SD1.5 directory; a LoRA cut is not a base (its hash needs one)."""
        sc = self.fresh()
        sc.render(spec_mutate=lambda d: d["artifacts"]["base"].update(uri=f"file:{sc.cuts[0]}"))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc)


# ---------------------------------------------------------------------------
# EXT-122: data binding
# ---------------------------------------------------------------------------

class TestDataBinding(DdpoCase):

    def test_train_file_outside_the_mounts(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(train_file="/elsewhere/train.jsonl"))
        self.refused("EXT-DATA", "artifacts.prompts.uri", sc=sc, text="train_file")

    def test_train_file_hash_differs(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(train_file_sha256="sha256:" + "1" * 64))
        self.refused("EXT-DATA", "artifacts.prompts.uri", sc=sc, text="train_file_sha256")

    def test_two_shards_refused(self):
        sc = self.fresh()
        shutil.copyfile(sc.data / "train.jsonl", sc.data / "train-b.jsonl")
        self.refused("EXT-DATA", "artifacts.prompts.uri", sc=sc, text="shard")


# ---------------------------------------------------------------------------
# EXT-123: observed consumption log (MM-24, MM-27)
# ---------------------------------------------------------------------------

class TestConsumptionLog(DdpoCase):

    def test_refusals(self):
        def short_epoch(samples, steps):                     # the sample and its step entry gone
            gone = [s for s in samples if s["epoch"] == 2 and s["sample_index"] == 0][0]["sample_id"]
            for st in steps:
                if gone in st["sample_ids"]:
                    st["sample_ids"].remove(gone)
            return [s for s in samples if s["sample_id"] != gone], steps
        def twice(samples, steps):
            steps[1]["sample_ids"].append(steps[0]["sample_ids"][0])
            return samples, steps
        def nowhere(samples, steps):
            steps[0]["sample_ids"].pop()
            return samples, steps
        def other_epoch(samples, steps):
            sid = steps[2]["sample_ids"].pop()                           # an epoch-2 sample in an epoch-1 step
            steps[0]["sample_ids"].append(sid)
            return samples, steps
        def gap(samples, steps):
            for st in steps[3:]:
                st["step"] += 1
            return samples, steps
        def prompt_text(samples, steps):
            samples[0]["prompt"] = "a different prompt"
            return samples, steps
        def unknown_prompt(samples, steps):
            samples[0]["prompt_id"] = "inet-9999"
            return samples, steps
        def bad_seed(samples, steps):
            samples[0]["seed"] = "12"
            return samples, steps
        for fn, text in ((short_epoch, "samples_per_epoch"), (twice, "step"), (nowhere, "step"),
                         (other_epoch, "a sample of epoch"),
                         (gap, "consecutive"), (prompt_text, "prompt"), (unknown_prompt, "inet-9999"),
                         (bad_seed, "seed")):
            with self.subTest(case=fn.__name__):
                sc = self.fresh()
                samples = read_jsonl(sc.stage_out / "samples.jsonl")
                steps = read_jsonl(sc.stage_out / "train_steps.jsonl")
                samples, steps = fn(samples, steps)
                write_jsonl(sc.stage_out / "samples.jsonl", samples)
                write_jsonl(sc.stage_out / "train_steps.jsonl", steps)
                self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text=text)

    def test_cut_counts_disagree(self):
        """MM-27: a cut's global_step and samples_consumed must equal the steps and samples logged up to it."""
        sc = self.fresh()
        cut = sc.cuts[0]
        sc.edit_json(cut / "cut.json", lambda d: d.update(samples_consumed=d["samples_consumed"] + 1))
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="samples_consumed")
        sc = self.fresh()
        cut = sc.cuts[0]
        sc.edit_json(cut / "cut.json", lambda d: d.update(global_step=d["global_step"] + 1))
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="global_step")

    def test_missing_logs_refused(self):
        for name in ("samples.jsonl", "train_steps.jsonl"):
            with self.subTest(log=name):
                sc = self.fresh()
                (sc.stage_out / name).unlink()
                self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text=name)

    def test_records_after_the_last_cut_orphaned(self):
        """Records of epochs after the given last cut are not consumption of these cuts (MM-27): the ledger covers
        the samples of steps up to the last cut's global step only."""
        sc = self.fresh()
        a = adapter()
        cuts = [(4, sc.cuts[0])]                                          # only cut epoch 2 (global step 4)
        self.assertEqual(a.consumption_problems(sc.stage_out, sc.data, cuts), [])
        cons = a.consumption(sc.stage_out, sc.data, cuts, run_id="r")
        self.assertEqual(cons["precision"], "row")
        self.assertEqual(sorted({r["step"] for r in cons["rows"]}), [1, 2, 3, 4])
        self.assertEqual(len(cons["rows"]), 16)


# ---------------------------------------------------------------------------
# EXT-124: exact overlap (MM-11)
# ---------------------------------------------------------------------------

class TestOverlap(DdpoCase):

    def test_overlap_refused(self):
        sc = self.fresh(overlap=2)
        self.refused("EXT-CONTAMINATION", "registration.contamination.allow_exact_overlap", sc=sc, text="2 item(s)")

    def test_allowed_overlap_reported(self):
        sc = self.fresh(overlap=2, reg_mutate=lambda r: r["contamination"].update(allow_exact_overlap=True))
        rep = json.loads((self.passes(sc) / "report" / "gate_report.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["contamination"], {"exact_overlap": 2, "allowed": True})

    def test_training_payloads(self):
        """The texts: every shard prompt and every sampled prompt up to the last cut."""
        sc = self.fresh()
        texts, why = adapter().training_payloads(sc.stage_out, sc.base, [(6, sc.final_cut)], data=sc.data)
        self.assertIsNone(why)
        shard = {r["prompt"] for r in read_jsonl(sc.data / "train.jsonl")}
        self.assertTrue(shard <= texts)
        self.assertEqual(texts, shard | {s["prompt"] for s in read_jsonl(sc.stage_out / "samples.jsonl")})


# ---------------------------------------------------------------------------
# EXT-125: seed, pins, profile, last cut, script, sampler
# ---------------------------------------------------------------------------

class TestRunChecks(DdpoCase):

    def test_seed(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(seed=1))
        self.refused("EXT-LAUNCH", "--launch-record.training_seed", sc=sc)

    def test_cut_seed(self):
        sc = self.fresh()
        sc.edit_json(sc.final_cut / "cut.json", lambda d: d.update(seed=1))
        self.refused("EXT-LAUNCH", "--launch-record.training_seed", sc=sc, text="cut.json")

    def test_framework_version(self):
        sc = self.fresh()
        p = sc.stage_out / "env-freeze.txt"
        p.write_text(p.read_text().replace("diffusers==0.42.1", "diffusers==0.41.0"))
        self.refused("EXT-FREEZE", "--stage-out", sc=sc)

    def test_pins_and_reference_line(self):
        for drop, text in (("peft==", "peft"), ("# ddpo algorithm reference", "ddpo-pytorch")):
            with self.subTest(drop=drop):
                sc = self.fresh()
                p = sc.stage_out / "env-freeze.txt"
                p.write_text("".join(l + "\n" for l in p.read_text().splitlines() if not l.startswith(drop)))
                self.refused("EXT-FREEZE", "--stage-out", sc=sc, text=text)

    def test_profile(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(profile="smoke"))
        self.refused("EXT-PROFILE", "--launch-record.profile", sc=sc)

    def test_last_cut(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["settings"].update(epochs=4))
        self.refused("EXT-STAGE", "--stage-out", sc=sc, text="epochs")

    def test_script(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "env-runtime.json", lambda d: d.update(script_sha256="sha256:" + "2" * 64))
        self.refused("EXT-LAUNCH", "--launch-record.script_sha256", sc=sc)
        sc = self.fresh()
        sc.edit_json(sc.final_cut / "cut.json", lambda d: d.update(script_sha256="sha256:" + "2" * 64))
        self.refused("EXT-LAUNCH", "--launch-record.script_sha256", sc=sc, text="cut.json")

    def test_sampler_code(self):
        sc = self.fresh()
        sc.edit_json(sc.final_cut / "cut.json", lambda d: d.update(sampler_code_sha256="sha256:" + "3" * 64))
        self.refused("EXT-SAMPLER", "--stage-out", sc=sc, text="sampler_code_sha256")

    def test_cut_sampler_not_registered(self):
        sc = self.fresh(reg_mutate=lambda r: r["sampler"].update(num_steps=50))
        self.refused("EXT-SAMPLER", "registration.sampler", sc=sc)


# ---------------------------------------------------------------------------
# EXT-126: ext-arm-fill on the evaluator's arm-<metric> directories
# ---------------------------------------------------------------------------

class TestArmFill(DdpoCase):

    def arms(self):
        sc = self.fresh(filled=False)
        return sc

    def meta(self, arm: Path) -> dict:
        return json.loads((arm / "eval_meta.json").read_text(encoding="utf-8"))

    def errs(self, proc) -> list:
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        return parse_errors(proc.stderr)

    def test_baseline_arm(self):
        """The base alone: model_hash = the base's manifest hash; the sampler hash the evaluator wrote (compact
        sorted JSON of the same sampler) is converted to the RFC 8785 hash."""
        sc = self.arms()
        arm = sc.base_eval / f"arm-{METRIC}"
        self.assertEqual(self.meta(arm)["sampler_hash"], compact_hash(SAMPLER))
        self.assertNotEqual(compact_hash(SAMPLER), sha_of(SAMPLER))
        items_before = (arm / "items.jsonl").read_bytes()
        proc = sc.fill(arm, sc.base, base=sc.base)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        meta = self.meta(arm)
        self.assertEqual(meta["checkpoint"], {"model_hash": sc.base_model, "base_model_hash": sc.base_model})
        self.assertEqual(meta["item_set_hash"], sc.item_set_hash)
        self.assertEqual(meta["sampler_hash"], sha_of(SAMPLER))
        self.assertIn("converted sampler_hash", proc.stdout)
        self.assertEqual((arm / "items.jsonl").read_bytes(), items_before)
        once = (arm / "eval_meta.json").read_bytes()
        proc = sc.fill(arm, sc.base, base=sc.base)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual((arm / "eval_meta.json").read_bytes(), once)

    def test_candidate_arm_needs_the_base(self):
        sc = self.arms()
        arm = sc.cand_eval / f"arm-{METRIC}"
        before = (arm / "eval_meta.json").read_bytes()
        errs = self.errs(sc.fill(arm, sc.final_cut))
        self.assertTrue([e for e in errs if e["field"] == "--base"], errs)
        self.assertEqual((arm / "eval_meta.json").read_bytes(), before)
        proc = sc.fill(arm, sc.final_cut, base=sc.base)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual(self.meta(arm)["checkpoint"], {"model_hash": sc.cand_model, "base_model_hash": sc.base_model})

    def test_supporting_arm(self):
        sc = self.arms()
        arm = sc.cand_eval / f"arm-{SUPPORT_METRIC}"
        proc = sc.fill(arm, sc.final_cut, base=sc.base)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual(self.meta(arm)["evaluator_id"], SUPPORT_ID)

    def test_other_sampler_hash_refused(self):
        sc = self.arms()
        arm = sc.base_eval / f"arm-{METRIC}"
        sc.edit_json(arm / "eval_meta.json", lambda m: m.update(sampler_hash="sha256:" + "5" * 64))
        before = (arm / "eval_meta.json").read_bytes()
        errs = self.errs(sc.fill(arm, sc.base, base=sc.base))
        self.assertTrue([e for e in errs if e["field"] == "ARM.sampler_hash"], errs)
        self.assertEqual((arm / "eval_meta.json").read_bytes(), before)

    def test_base_model_hash_mismatch_refused(self):
        sc = self.arms()
        arm = sc.cand_eval / f"arm-{METRIC}"
        sc.edit_json(arm / "eval_meta.json", lambda m: m["checkpoint"].update(base_model_hash="sha256:" + "6" * 64))
        errs = self.errs(sc.fill(arm, sc.final_cut, base=sc.base))
        self.assertTrue([e for e in errs if e["field"] == "ARM.checkpoint.base_model_hash"], errs)

    def test_base_must_be_a_base_directory(self):
        """`--base` names the plain base directory, never a cut (a cut's hash would compose nothing)."""
        sc = self.arms()
        arm = sc.cand_eval / f"arm-{METRIC}"
        before = (arm / "eval_meta.json").read_bytes()
        for bad in (sc.cuts[0], sc.data):
            with self.subTest(base=bad.name):
                errs = self.errs(sc.fill(arm, sc.final_cut, base=bad))
                self.assertTrue([e for e in errs if e["field"] == "--base"], errs)
        self.assertEqual((arm / "eval_meta.json").read_bytes(), before)

    def test_fill_then_collect(self):
        """The operator's sequence: eval_images.py arms, ext-arm-fill on each, then ext-collect."""
        sc = self.arms()
        for arm, ck in ((sc.baseline, sc.base), (sc.candidate, sc.final_cut)):
            proc = sc.fill(arm, ck, base=sc.base)
            self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.passes(sc)


# ---------------------------------------------------------------------------
# EXT-127: registration (sampler keys, construct alignment)
# ---------------------------------------------------------------------------

class TestRegistration(unittest.TestCase):

    def reg(self=None) -> dict:
        text = (DDPO / "registration.template.yaml").read_text(encoding="utf-8")
        text = text.replace("{item_set_hash}", "sha256:" + "a" * 64).replace("{sampler}", json.dumps(SAMPLER))
        return yaml.safe_load(text.replace("{evaluator_hash}", "sha256:" + "b" * 64))

    def fields(self, reg: dict, evaluator: str = EVALUATOR_ID) -> list[str]:
        from gatedtrain.registration import check_contents
        return [p.field for p in check_contents(reg, evaluator_id=evaluator)]

    def test_template_accepted(self):
        self.assertEqual(self.fields(self.reg()), [])

    def test_sampler_keys(self):
        reg = self.reg()
        reg["sampler"]["temperature"] = 0.7
        del reg["sampler"]["num_steps"]
        f = self.fields(reg)
        self.assertIn("registration.sampler.temperature", f)
        self.assertIn("registration.sampler.num_steps", f)

    def test_construct_alignment_required(self):
        reg = self.reg()
        del reg["construct_alignment"]
        self.assertIn("registration.construct_alignment", self.fields(reg))
        reg["construct_alignment"] = "  "
        self.assertIn("registration.construct_alignment", self.fields(reg))

    def test_not_required_for_the_t0_arm(self):
        reg = self.reg()
        del reg["construct_alignment"]
        reg["evaluators"] = [{"id": SUPPORT_ID, "manifest_hash": "sha256:" + "c" * 64}]
        reg["clause"]["metric"] = SUPPORT_METRIC
        self.assertNotIn("registration.construct_alignment", self.fields(reg, SUPPORT_ID))

    def test_collect_refuses_without_it(self):
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ddpo-reg-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        sc = DdpoScenario(tmp / "sc", n_items=12, reg_mutate=lambda r: r.pop("construct_alignment"))
        proc = sc.collect(tmp / "runs")
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn("registration.construct_alignment", [e["field"] for e in parse_errors(proc.stderr)])


# ---------------------------------------------------------------------------
# EXT-129: launch (host rows, GPU request) and EXT-130: smoke on the deciding metric only
# ---------------------------------------------------------------------------

class TestLaunch(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ddpo-launch-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.sc = DdpoScenario(self.tmp / "sc", n_items=12)
        self.out = self.sc.output_root / "area3-sd15" / "full-2"
        self.out.mkdir()

    def request(self, **overrides) -> dict:
        rec = self.sc.launch_record()
        req = {"area": 2, "label": "mm-a3-sd15-full", "host_work": rec["host_work"],
               "runtime": {"kind": "venv_in_container", "image": IMAGE_ID, "venv_lock": str(self.sc.runtime / "uv.lock")},
               "framework": rec["framework"], "training_seed": 3000, "gpus": True, "network": "none",
               "network_reason": None, "user": run_user(),
               "env": {"HF_HUB_OFFLINE": "1", "HOME": "/tmp/home", "XDG_CACHE_HOME": "/tmp/cache",
                       "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"},
               "mounts": [m if m["mode"] == "ro" else {**m, "host": str(self.out)} for m in rec["mounts"]],
               "tmpfs": ["/tmp"], "output_dir": str(self.out), "gpu_est_h": 0.15, "argv": rec["argv"]}
        req.update(overrides)
        return req

    def launch(self, req: dict, *, profile: str = "smoke", smoke: Path | None = None, extra: list[str] = ()):
        path = self.tmp / f"request-{len(list(self.tmp.glob('request-*')))}.yaml"
        path.write_text(yaml.safe_dump(req, sort_keys=False), encoding="utf-8")
        out = path.with_suffix(".json")
        args = ["ext-launch", "--adapter", ADAPTER, "--profile", profile, "--request", str(path), "--gpu-log",
                str(self.sc.host_work / "gpu-log.md"), "--out-record", str(out)]
        if smoke is not None:
            args += ["--smoke-record", str(smoke)]
        return run_cli(args + list(extra), cwd=self.tmp, env=self.sc.env()), out

    def refused(self, proc, out, rule, field):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        self.assertTrue([e for e in errs if e["rule"] == rule and e["field"] == field], f"{rule} {field}: {errs}")
        self.assertFalse(out.exists())

    def test_gpu_request_accepted(self):
        proc, out = self.launch(self.request())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn("--runtime=nvidia", proc.stdout)
        self.assertIn("GATEDTRAIN_GPU_STEP_EST_H=0.15 ", proc.stdout)
        rec = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual((rec["adapter"], rec["framework"]["name"]), (ADAPTER, "diffusers"))

    def test_optional_allocator_name_allowed(self):
        req = self.request()
        req["env"]["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        proc, out = self.launch(req)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_other_env_name_refused(self):
        req = self.request()
        req["env"]["HF_HOME"] = "/tmp/hf"
        proc, out = self.launch(req)
        self.refused(proc, out, "EXT-LAUNCH", "--request.env.HF_HOME")

    def test_spec_optional_and_checked(self):
        proc, out = self.launch(self.request(), extra=["--spec", str(self.sc.paths["spec"])])
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        req = self.request()
        req["mounts"] = [m for m in req["mounts"] if m["mode"] == "rw"]
        proc, out = self.launch(req, extra=["--spec", str(self.sc.paths["spec"])])
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        fields = [e["field"] for e in parse_errors(proc.stderr)]
        self.assertIn("artifacts.base.uri", fields)
        self.assertIn("artifacts.prompts.uri", fields)

    def test_registry_host_rows(self):
        reg = registry()
        from gatedtrain.ext.rowkit import python_tree
        from gatedtrain_recipe_ddpo_sd15.rows import OUT_NAME, RUNTIME
        tree = python_tree(RUNTIME)
        self.assertEqual(reg.input_roots(ADAPTER), ((str(Path(tree).parent),) if tree else ())
                         + (RUNTIME, "{output_root}/" + OUT_NAME))
        self.assertEqual(set(reg.env_names(ADAPTER)), set(ENV_NAMES) | {"PYTORCH_CUDA_ALLOC_CONF"})
        self.assertEqual(reg.env_paths(ADAPTER), ("HOME", "XDG_CACHE_HOME"))
        self.assertEqual(reg.scan_skip(ADAPTER)[0], f"{RUNTIME}/.venv")
        self.assertEqual(reg.interpreter(ADAPTER), f"{RUNTIME}/.venv/bin/python")
        self.assertEqual(reg.interpreter_trees(ADAPTER), (tree,) if tree else ())

    def test_full_run_smoke_on_the_deciding_metric(self):
        """EXT-130 (MM-38, decision 5): the smoke is judged on webp_lossless_bpp only: [0, null] has no mean rule,
        the mode and sd rules apply; the binary constraint metric is not judged (an all-zero violation baseline
        does not stop the area)."""
        good = write_canonical(self.tmp / "smoke.json", smoke_record(base_model=self.sc.base_model,
                                                                     item_set=self.sc.item_set_hash))
        proc, out = self.launch(self.request(), profile="full", smoke=good)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn("GATEDTRAIN_GPU_STEP_EST_H=1.3 ", proc.stdout)
        flat = write_canonical(self.tmp / "smoke-flat.json", smoke_record(
            base_model=self.sc.base_model, item_set=self.sc.item_set_hash, baseline_scores=[9.0] * 10 + [8.5] * 10))
        proc, out = self.launch(self.request(), profile="full", smoke=flat)
        self.refused(proc, out, "EXT-SMOKE", "--smoke-record")
        self.assertIn("degenerate_mode", proc.stderr)



# ---------------------------------------------------------------------------
# Review round 8 (tests/README.md section 11.20, EXT-131 to EXT-137)
# ---------------------------------------------------------------------------

class TestUnhashableBase(DdpoCase):
    """EXT-131 (review M1): a base that cannot be hashed refuses (exit 3, EXT-BASE), never a traceback (exit 1)."""

    def test_symlink_in_base(self):
        sc = self.fresh()
        (sc.base / "extra.json").symlink_to(sc.base / "model_index.json")
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc)

    def test_missing_base(self):
        sc = self.fresh()
        sc.render(spec_mutate=lambda d: d["artifacts"]["base"].update(uri=f"file:{sc.root / 'no-such-base'}"))
        hits = self.refused("EXT-BASE", "artifacts.base.uri", sc=sc)
        self.assertTrue(hits)


class TestDerivedFields(DdpoCase):
    """EXT-132 (review M2): every field train_ddpo.py derives is re-derived (train_ddpo.py:595-625, :671-672)."""

    def edit(self, sc, fn):
        samples = read_jsonl(sc.stage_out / "samples.jsonl")
        steps = read_jsonl(sc.stage_out / "train_steps.jsonl")
        fn(samples, steps)
        write_jsonl(sc.stage_out / "samples.jsonl", samples)
        write_jsonl(sc.stage_out / "train_steps.jsonl", steps)

    @staticmethod
    def reid(x, steps, **changes):
        """Change fields of a sample line and re-derive its run sample_id (and the step reference), so only the changed
        field is wrong."""
        old = x["sample_id"]
        x.update(changes)
        x["sample_id"] = compact_hash(["ddpo-sd15", 3000, x["attempt"], x["epoch"], x["sample_index"], x["prompt_id"],
                                       x["seed"], x["member"]])
        for st in steps:
            st["sample_ids"] = [x["sample_id"] if s == old else s for s in st["sample_ids"]]

    def test_seed(self):
        sc = self.fresh()
        self.edit(sc, lambda samples, steps: self.reid(samples[0], steps, seed=12345))
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="derive_seed")

    def test_sample_id(self):
        sc = self.fresh()
        def fn(samples, steps):
            old = samples[0]["sample_id"]
            samples[0]["sample_id"] = "sha256:" + "7" * 64
            for st in steps:
                st["sample_ids"] = [samples[0]["sample_id"] if s == old else s for s in st["sample_ids"]]
        self.edit(sc, fn)
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="sample_id is not")

    def test_member(self):
        sc = self.fresh()
        self.edit(sc, lambda samples, steps: self.reid(samples[0], steps, member=77))
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="images_per_prompt")

    def test_reward(self):
        sc = self.fresh()
        self.edit(sc, lambda samples, steps: samples[0].update(reward=samples[0]["reward"] + 1.0))
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="-jpeg_kb")

    def test_steps_range(self):
        sc = self.fresh()
        self.edit(sc, lambda samples, steps: samples[0].update(steps=[samples[0]["steps"][0],
                                                                      samples[0]["steps"][1] + 1]))
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="the epoch's steps")


class TestEvalBaseArgument(unittest.TestCase):
    """EXT-133 (review L1): eval_images.py loads `--base` without hashing it, so a ddpo request may name no base other
    than the script's default `<runtime>/models/stable-diffusion-v1-5` inside the read-only mount."""

    def test_rule(self):
        a = adapter()
        rt = "/srv/recipes/ddpo-sd15/runtime"
        ev = [f"{rt}/.venv/bin/python", f"{rt}/eval_images.py", "--base-only", "--out", "/out"]
        self.assertEqual(a.argv_problems(ev), [])
        self.assertEqual(a.argv_problems(ev + ["--base", f"{rt}/models/stable-diffusion-v1-5"]), [])
        for bad in (ev + ["--base", "/inputs/other-sd15"], ev + ["--base=/inputs/other-sd15"],
                    ev + ["--base", f"{rt}/models/stable-diffusion-v1-5/../other"]):
            with self.subTest(argv=bad[-1]):
                self.assertTrue(a.argv_problems(bad))
        tr = [f"{rt}/.venv/bin/python", f"{rt}/train_ddpo.py", "--base", "/inputs/x"]
        self.assertEqual(a.argv_problems(tr), [])         # training: collect binds the base (EXT-121)

    def test_launch_refuses(self):
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ddpo-evalbase-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        sc = DdpoScenario(tmp / "sc", n_items=12)
        out = sc.output_root / "area3-sd15" / "eval-1"
        out.mkdir()
        (sc.runtime / "models" / "other").mkdir()
        rec = sc.launch_record()
        req = {"area": 2, "label": "mm-a3-sd15-eval", "host_work": rec["host_work"],
               "runtime": {"kind": "venv_in_container", "image": IMAGE_ID, "venv_lock": str(sc.runtime / "uv.lock")},
               "framework": rec["framework"], "training_seed": 3000, "gpus": True, "network": "none",
               "network_reason": None, "user": run_user(),
               "env": {"HF_HUB_OFFLINE": "1", "HOME": "/tmp/home", "XDG_CACHE_HOME": "/tmp/cache"},
               "mounts": [rec["mounts"][0], {"host": str(out), "container": "/out", "mode": "rw"}],
               "tmpfs": ["/tmp"], "output_dir": str(out), "gpu_est_h": 0.03,
               "argv": [f"{RUNTIME_CONTAINER}/.venv/bin/python", f"{RUNTIME_CONTAINER}/eval_images.py", "--base-only",
                        "--items", f"{RUNTIME_CONTAINER}/items", "--out", "/out", "--eval-seeds", "11,12"]}
        def launch(r, name):
            path = tmp / f"{name}.yaml"
            path.write_text(yaml.safe_dump(r, sort_keys=False), encoding="utf-8")
            return run_cli(["ext-launch", "--adapter", ADAPTER, "--profile", "smoke", "--request", str(path),
                            "--gpu-log", str(sc.host_work / "gpu-log.md"), "--out-record", str(tmp / f"{name}.json")],
                           cwd=tmp, env=sc.env())
        proc = launch(req, "ok")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        req["argv"] = req["argv"] + ["--base", f"{RUNTIME_CONTAINER}/models/other"]
        proc = launch(req, "other")
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn("--request.argv", [e["field"] for e in parse_errors(proc.stderr)])


class TestStatementRules(unittest.TestCase):
    """EXT-134 (review L3): the statement is refused for an evaluator that is not construct-aligned; the report shows
    the registration's statement, and the manifest's beside it when they differ."""

    def test_statement_for_a_non_aligned_evaluator_refused(self):
        reg = TestRegistration.reg(None)
        reg["evaluators"] = [{"id": SUPPORT_ID, "manifest_hash": "sha256:" + "c" * 64}]
        reg["clause"]["metric"] = SUPPORT_METRIC
        from gatedtrain.registration import check_contents
        self.assertIn("registration.construct_alignment",
                      [p.field for p in check_contents(reg, evaluator_id=SUPPORT_ID)])

    def test_report_shows_both(self):
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ddpo-stmt-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        own = "A pass shows only that the JPEG gain carries over to WebP; it says nothing about image quality."
        sc = DdpoScenario(tmp / "sc", n_items=12, reg_mutate=lambda r: r.update(construct_alignment=own))
        proc = sc.collect(tmp / "runs")
        self.assertIn(proc.returncode, (EXIT_OK, 1), describe(proc))
        run = parse_run_dir(proc.stdout)
        rep = json.loads((run / "report" / "gate_report.json").read_text(encoding="utf-8"))
        ca = rep["construct_alignment"]
        self.assertEqual(ca["statement"], own)
        self.assertEqual(ca["manifest_statement"], registry().EVALUATOR_ROWS[EVALUATOR_ID]["construct_aligned"]["statement"])
        text = (run / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertIn(f"construct_alignment: {own}\n", text)
        self.assertIn("construct_alignment_manifest: ", text)

    def test_same_statement_printed_once(self):
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ddpo-stmt1-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        row = registry().EVALUATOR_ROWS[EVALUATOR_ID]["construct_aligned"]["statement"]
        sc = DdpoScenario(tmp / "sc", n_items=12, reg_mutate=lambda r: r.update(construct_alignment=row))
        proc = sc.collect(tmp / "runs")
        self.assertIn(proc.returncode, (EXIT_OK, 1), describe(proc))
        text = (parse_run_dir(proc.stdout) / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertIn(f"construct_alignment: {row}\n", text)
        self.assertNotIn("construct_alignment_manifest: ", text)


class TestBaseWalkedOnce(unittest.TestCase):
    """EXT-135 (review L4): identity of the base reuses the adapter's walk (2.1 GB on the real base)."""

    def test_one_walk(self):
        from unittest import mock
        import gatedtrain_recipe_ddpo_sd15.adapter as mod
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ddpo-walk-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        base = build_base(tmp / "base")
        cut = build_cut(tmp / "run", 2, global_step=4, samples_consumed=16, base=base, script_sha256="sha256:" + "1" * 64)
        a = adapter()
        real = mod.manifest_entries
        with mock.patch.object(mod, "manifest_entries", side_effect=real) as walk:
            a.identity(cut, base)                         # the cut and the base
            a.identity(base)                              # the base again (ext-arm-fill's base_model_hash)
            a.identity(base, base)
        walked = [Path(c.args[0]).resolve() for c in walk.call_args_list]
        self.assertEqual(walked.count(base.resolve()), 1, walked)


class TestInnerEpochs(DdpoCase):
    """EXT-136 (review L5): num_inner_epochs 2: every sample is applied once per inner epoch; one ledger record per
    (sample, applying step); a cut's samples_consumed counts distinct samples."""

    def test_two_inner_epochs(self):
        import pyarrow.parquet as pq
        sc = self.fresh(inner_epochs=2)
        steps = read_jsonl(sc.stage_out / "train_steps.jsonl")
        self.assertEqual(len(steps), 12)
        self.assertEqual(json.loads((sc.final_cut / "cut.json").read_text())["global_step"], 12)
        run = self.passes(sc)
        rows = pq.read_table(run / "ledger" / "consumption.parquet").to_pylist()
        samples = read_jsonl(sc.stage_out / "samples.jsonl")
        self.assertEqual(len(rows), 2 * len(samples))
        self.assertEqual(len({r["sample_id"] for r in rows}), len(rows))

    def test_inner_epoch_missing_a_sample(self):
        sc = self.fresh(inner_epochs=2)
        steps = read_jsonl(sc.stage_out / "train_steps.jsonl")
        moved = [st for st in steps if st["epoch"] == 1 and st["inner"] == 1][0]["sample_ids"].pop()
        write_jsonl(sc.stage_out / "train_steps.jsonl", steps)
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="inner 1")


class TestConvertedSampler(DdpoCase):
    """EXT-137 (review L5): fill converts only the arm's own sampler's encoding; a converted arm whose sampler is not
    the registered one is still refused by collect (the sb3_ppo case lives in its recipe repository)."""

    def test_converted_other_sampler_refused(self):
        sc = self.fresh(filled=False)
        other = {**SAMPLER, "num_steps": 50}
        for arm in (sc.baseline, sc.candidate):
            sc.edit_json(arm / "eval_meta.json", lambda m: m.update(sampler=other, sampler_hash=compact_hash(other)))
        for arm, ck in ((sc.baseline, sc.base), (sc.candidate, sc.final_cut)):
            proc = sc.fill(arm, ck, base=sc.base)
            self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
            self.assertIn("converted sampler_hash", proc.stdout)
            self.assertEqual(json.loads((arm / "eval_meta.json").read_text())["sampler_hash"], sha_of(other))
        hits = self.refused("EXT-ARM", "--eval-candidate", sc=sc)
        text = " ".join(h["message"] for h in hits)
        self.assertIn("sampler is", text)
        self.assertIn("sampler_hash is", text)


if __name__ == "__main__":
    unittest.main()
