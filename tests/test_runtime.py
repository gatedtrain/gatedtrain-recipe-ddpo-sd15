"""The runtime scripts that run inside the GPU container (runtime/train_ddpo.py, runtime/eval_images.py), on CPU and
without torch, diffusers or Pillow: their pure helpers and record writers, checked against what the adapter and
gatedtrain read from their outputs at collect.

Why: these scripts run only on a GPU, so a fault in a part that needs no GPU (seed derivation, the sample-id encoding,
the cut layout, the safetensors header, the settings, the per-prompt advantages, the shared sampler block, the item
split) was found only after GPU time. The adapter tests (test_adapter.py) build their runs with their own copy of the
runtime's logic; here the runtime's own functions write the files and the adapter reads them.

The scripts import only the standard library at module level (torch, diffusers, Pillow, numpy are imported inside the
functions that need them), so they load as they are; numpy comes with gatedtrain. Nothing in runtime/ is changed: its
SHA-256s are recorded in RESULT.md and eval_images.py's is the registered scorer pin (rows.EVAL_IMAGES_SHA256).
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import random
import shutil
import stat
import struct
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import yaml

from gatedtrain.ext.sdk import compact_sha256, derive_seed as sdk_derive_seed, safetensors_problem

from gatedtrain_recipe_ddpo_sd15 import adapter as A
from gatedtrain_recipe_ddpo_sd15 import rows as R

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "runtime"
GATE = REPO / "results" / "2026-11-13" / "gate"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = load_script(RUNTIME / "train_ddpo.py", "_ddpo_runtime_train")
E = load_script(RUNTIME / "eval_images.py", "_ddpo_runtime_eval")


def safetensors_file(path: Path, header: dict, data: bytes) -> None:
    """A safetensors file whose header keys keep the given (insertion) order, as safetensors' hash maps write it."""
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(text)) + text + data)


class FakeUnet:
    """`UNet2DConditionModel.save_lora_adapter` as far as save_cut uses it: a safetensors file, mode 0600, its header
    tensor table and LoraConfig metadata (target_modules a list of a set) in an arbitrary order."""

    def __init__(self, order_seed: int = 0):
        self.order_seed = order_seed

    def save_lora_adapter(self, save_directory, adapter_name, safe_serialization, weight_name):
        assert adapter_name == "default" and safe_serialization is True
        rnd = random.Random(self.order_seed)
        targets = ["to_k", "to_q", "to_v", "to_out.0"]
        rnd.shuffle(targets)
        names = [f"down_blocks.0.attentions.0.transformer_blocks.0.attn1.{t}.lora_{ab}.weight"
                 for t in ("to_q", "to_k") for ab in ("A", "B")]
        tensors = {n: {"dtype": "F32", "shape": [2], "data_offsets": [8 * i, 8 * i + 8]} for i, n in enumerate(names)}
        keys = list(tensors)
        rnd.shuffle(keys)
        meta = {"format": "pt", "lora_adapter_metadata": json.dumps({"r": 4, "lora_alpha": 4, "target_modules": targets})}
        header = {"__metadata__": meta, **{k: tensors[k] for k in keys}}
        path = Path(save_directory) / weight_name
        safetensors_file(path, header, bytes(range(8 * len(names))))
        os.chmod(path, 0o600)


def plain_base(d: Path) -> dict:
    """A tiny plain base directory and the MODELS.json record train_ddpo.py binds it with."""
    files = {"model_index.json": b"{}\n", "scheduler/scheduler_config.json": b'{"x": 1}\n',
             "unet/diffusion_pytorch_model.fp16.safetensors": b"unet weights"}
    for rel, data in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_bytes(data)
    return {"repo": "stable-diffusion-v1-5/stable-diffusion-v1-5", "revision": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
            "files": {rel: {"sha256": T.sha256_bytes(data), "bytes": len(data)} for rel, data in files.items()}}


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ddpo-runtime-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def patch(self, obj, name, value):
        old = getattr(obj, name)
        setattr(obj, name, value)
        self.addCleanup(setattr, obj, name, old)

    def models_json(self, base: Path) -> Path:
        """MODELS.json for `base`, patched into both scripts (they read HERE/models/MODELS.json)."""
        path = self.tmp / "MODELS.json"
        path.write_text(json.dumps({"models": {T.BASE_NAME: plain_base(base)}}), encoding="utf-8")
        self.patch(T, "MODELS_JSON", path)
        self.patch(E, "MODELS_JSON", path)
        return path


# ---------------------------------------------------------------------------
# Seeds and hashes the adapter re-derives
# ---------------------------------------------------------------------------

class TestSeedsAndIds(unittest.TestCase):
    def test_derive_seed_is_the_adapters(self):
        for parts in [("sample", 3000, 1, 0), ("sample", 3000, 24, 63), ("prompts", 3000, 1), ("perm", 7, 2, 0),
                      ("eval", 11, "inet-0007"), ("torch", 0)]:
            s = T.derive_seed(*parts)
            self.assertEqual(s, sdk_derive_seed(*parts))
            self.assertEqual(s, E.derive_seed(*parts))
            self.assertTrue(0 <= s < A.MAX_SEED)
        self.assertNotEqual(T.derive_seed("sample", 3000, 1, 0), T.derive_seed("sample", 3000, 0, 1))

    def test_sample_id_encoding_is_the_adapters(self):
        # train_ddpo.py: sha256_bytes(canonical([...])); adapter: compact_sha256([...]) (non-ASCII kept as UTF-8).
        for parts in (["ddpo-sd15", 3000, 1, 1, 0, "inet-0001", 2 ** 62 + 5, 0],
                      ["ddpo-sd15", 3000, 2, 3, 7, "chéri ü", 12, 3]):
            self.assertEqual(T.sha256_bytes(T.canonical(parts)), compact_sha256(parts))

    def test_shared_sampler_block_is_byte_identical(self):
        # eval_images.py check_cut refuses a cut whose sampler_code_sha256 is not its own; the block must not drift.
        def block(path):
            text = path.read_text(encoding="utf-8")
            return text[text.index("# BEGIN SHARED SAMPLER"):text.index("# END SHARED SAMPLER")]

        self.assertEqual(block(RUNTIME / "train_ddpo.py"), block(RUNTIME / "eval_images.py"))
        self.assertEqual(T.sampler_code_sha256(), E.sampler_code_sha256())
        # The published cuts and runs record this value (tests/_ddpo.py SAMPLER_CODE_SHA, RESULT.md: unchanged).
        self.assertEqual(T.sampler_code_sha256(),
                         "sha256:7707a4ea20320ef76d97bfa484ed24d8017446a8e7b046017a30c4c1cdf7211c")

    def test_eval_script_is_the_registered_scorer(self):
        self.assertEqual(T.sha256_file(RUNTIME / "eval_images.py"), R.EVAL_IMAGES_SHA256)


# ---------------------------------------------------------------------------
# Settings (pinned: README "Changes against ddpo-pytorch", "Profiles and cost")
# ---------------------------------------------------------------------------

class TestSettings(unittest.TestCase):
    def settings(self, *argv):
        return T.resolve_settings(T.parse_args(["--out", "/tmp/x", *argv]))

    def test_common_settings_are_the_recorded_ones(self):
        s = self.settings("--profile", "full")
        want = {"num_steps": 20, "eta": 1.0, "guidance_scale": 5.0, "resolution": 512, "learning_rate": 3.0e-4,
                "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_weight_decay": 1.0e-4, "adam_epsilon": 1.0e-8,
                "max_grad_norm": 1.0, "adv_clip_max": 5.0, "clip_range": 1.0e-4, "timestep_fraction": 1.0,
                "train_cfg": True, "num_inner_epochs": 1, "stat_buffer_size": 16, "stat_min_count": 4,
                "reward": "jpeg_compressibility", "jpeg_quality": 95, "precision": "fp16"}
        self.assertEqual({k: s[k] for k in want}, want)
        self.assertEqual(T.LORA, {"r": 4, "lora_alpha": 4, "init_lora_weights": "gaussian",
                                  "target_modules": ["to_k", "to_q", "to_v", "to_out.0"]})

    def test_profiles(self):
        full = self.settings("--profile", "full")
        self.assertEqual((full["epochs"], full["prompts_per_epoch"], full["images_per_prompt"], full["sample_batch"],
                          full["train_batch"], full["updates_per_epoch"], full["save_every"]), (24, 16, 4, 8, 2, 2, 4))
        self.assertEqual((full["samples_per_epoch"], full["grad_accum_batches"], full["num_train_timesteps"],
                          full["seed"]), (64, 16, 20, 3000))
        smoke = self.settings("--profile", "smoke")
        self.assertEqual((smoke["epochs"], smoke["samples_per_epoch"], smoke["save_every"]), (2, 64, 1))
        cpu = self.settings("--profile", "cpu-sanity")
        self.assertEqual((cpu["num_steps"], cpu["resolution"], cpu["precision"], cpu["samples_per_epoch"],
                          cpu["grad_accum_batches"]), (2, 256, "fp32", 4, 2))
        # The result's run: --epochs 12 (RESULT.md; decision 6) and the memory knobs.
        o = self.settings("--profile", "full", "--epochs", "12", "--train-batch", "4", "--no-gradient-checkpointing",
                          "--seed", "5")
        self.assertEqual((o["epochs"], o["train_batch"], o["gradient_checkpointing"], o["seed"], o["grad_accum_batches"]),
                         (12, 4, False, 5, 8))

    def test_every_profile_gives_the_adapter_its_settings(self):
        # adapter._ledger_plan refuses a run_args.json without these positive integers.
        for profile in T.PROFILES:
            s = self.settings("--profile", profile)
            for k in ("samples_per_epoch", "updates_per_epoch", "num_inner_epochs", "images_per_prompt", "epochs"):
                self.assertTrue(isinstance(s[k], int) and s[k] > 0, (profile, k))
            self.assertEqual(s["samples_per_epoch"],
                             s["train_batch"] * s["updates_per_epoch"] * s["grad_accum_batches"], profile)
            self.assertEqual(s["samples_per_epoch"] % s["sample_batch"], 0, profile)

    def test_indivisible_batches_are_refused(self):
        with self.assertRaises(SystemExit):
            self.settings("--profile", "full", "--train-batch", "3")
        with self.assertRaises(SystemExit):
            self.settings("--profile", "full", "--train-batch", "64")     # 64 % (64 x 2) != 0
        odd = {**T.PROFILES["smoke"], "sample_batch": 6}                 # 64 % 6 != 0, 64 % (2 x 2) == 0
        with unittest.mock.patch.dict(T.PROFILES, {"smoke": odd}), self.assertRaises(SystemExit):
            self.settings("--profile", "smoke")
        import contextlib
        import io

        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            T.parse_args(["--profile", "huge", "--out", "x"])

    def test_deployment_sampler_is_the_registered_one(self):
        # The cuts' sampler.json (train sampler_doc at COMMON) is the sampler the gate registered and the evaluator's
        # default (EXT-125, EXT-127).
        reg = yaml.safe_load((GATE / "registration-gatedtrain.yaml").read_text(encoding="utf-8"))["sampler"]
        c = T.COMMON
        doc = T.sampler_doc(c["num_steps"], c["eta"], c["guidance_scale"], c["resolution"], reg["scheduler_config"])
        self.assertEqual(doc, reg)
        self.assertEqual(sorted(doc), sorted(R.EVALUATOR_ROWS["t2i.sd15_webp_items@0.1"]["sampler_keys"]))
        self.assertEqual(E.SAMPLER_DEFAULT, {k: c[k] for k in ("num_steps", "eta", "guidance_scale", "resolution")})

    def test_sampler_doc_sorts_use_default_values(self):
        a = T.sampler_doc(20, 1.0, 5.0, 512, {"_use_default_values": ["b", "a", "c"], "x": 1})
        b = E.sampler_doc(20, 1.0, 5.0, 512, {"x": 1, "_use_default_values": ["c", "b", "a"]})
        self.assertEqual(T.canonical(a), T.canonical(b))
        self.assertEqual(a["scheduler_config"]["_use_default_values"], ["a", "b", "c"])


# ---------------------------------------------------------------------------
# Per-prompt advantages (ddpo-pytorch stat_tracking.py)
# ---------------------------------------------------------------------------

class TestStatTracker(unittest.TestCase):
    def test_below_min_count_uses_the_batch_statistics(self):
        import numpy as np

        tr = T.PerPromptStatTracker(buffer_size=16, min_count=4)
        prompts = ["a", "a", "b", "b"]
        rewards = [-40.0, -42.0, -30.0, -36.0]
        adv = tr.update(prompts, rewards)
        r = np.array(rewards)
        np.testing.assert_allclose(adv, (r - r.mean()) / (r.std() + 1e-6))

    def test_at_min_count_uses_the_prompts_buffer(self):
        import numpy as np

        tr = T.PerPromptStatTracker(buffer_size=16, min_count=4)
        rewards = [-40.0, -42.0, -41.0, -45.0, -10.0, -12.0, -11.0, -15.0]
        adv = tr.update(["a"] * 4 + ["b"] * 4, rewards)
        for k in (slice(0, 4), slice(4, 8)):
            r = np.array(rewards[k])
            np.testing.assert_allclose(adv[k], (r - r.mean()) / (r.std() + 1e-6))
        # A second epoch: the buffer holds both epochs' rewards of the prompt.
        adv2 = tr.update(["a"] * 4, [-30.0, -31.0, -32.0, -33.0])
        buf = np.array(rewards[:4] + [-30.0, -31.0, -32.0, -33.0])
        np.testing.assert_allclose(adv2, (np.array([-30.0, -31.0, -32.0, -33.0]) - buf.mean()) / (buf.std() + 1e-6))

    def test_buffer_is_bounded_and_equal_rewards_stay_finite(self):
        import numpy as np

        tr = T.PerPromptStatTracker(buffer_size=16, min_count=4)
        for e in range(6):
            adv = tr.update(["a"] * 4, [-20.0] * 4)
            self.assertTrue(np.all(np.isfinite(adv)))
            np.testing.assert_allclose(adv, 0.0)
        self.assertEqual(len(tr.stats["a"]), 16)

    def test_stats_serialize_into_trainer_state(self):
        # save_cut writes the tracker (numpy string keys and float64 values) into trainer_state.json.
        tr = T.PerPromptStatTracker(16, 4)
        tr.update(["inet-0001", "inet-0002"], [-1.5, -2.5])
        doc = {k: [float(x) for x in v] for k, v in sorted(tr.stats.items())}
        self.assertEqual(json.loads(json.dumps(doc)), {"inet-0001": [-1.5], "inet-0002": [-2.5]})


# ---------------------------------------------------------------------------
# Cuts (save_cut, canonicalize_safetensors_header) as the adapter and eval_images.py check them
# ---------------------------------------------------------------------------

class TestSafetensorsHeader(Tmp):
    def test_canonical_header_is_order_independent_and_keeps_the_tensors(self):
        outs = []
        for seed in (1, 2, 3):
            d = self.tmp / str(seed)
            d.mkdir()
            FakeUnet(seed).save_lora_adapter(d, "default", True, T.LORA_WEIGHTS)
            raw = (d / T.LORA_WEIGHTS).read_bytes()
            n = struct.unpack("<Q", raw[:8])[0]
            T.canonicalize_safetensors_header(d / T.LORA_WEIGHTS)
            new = (d / T.LORA_WEIGHTS).read_bytes()
            m = struct.unpack("<Q", new[:8])[0]
            self.assertEqual(m % 8, 0)
            self.assertEqual(new[8 + m:], raw[8 + n:])                 # tensor bytes and offsets unchanged
            self.assertEqual(json.loads(new[8:8 + m]), {**json.loads(raw[8:8 + n]), "__metadata__":
                                                        json.loads(new[8:8 + m])["__metadata__"]})
            meta = json.loads(json.loads(new[8:8 + m])["__metadata__"]["lora_adapter_metadata"])
            self.assertEqual(meta["target_modules"], ["to_k", "to_out.0", "to_q", "to_v"])
            self.assertIsNone(safetensors_problem(d / T.LORA_WEIGHTS))
            T.canonicalize_safetensors_header(d / T.LORA_WEIGHTS)       # idempotent
            self.assertEqual((d / T.LORA_WEIGHTS).read_bytes(), new)
            outs.append(new)
        self.assertEqual(len(set(outs)), 1)


class TestSaveCut(Tmp):
    def cut(self, epoch=2, global_step=4, consumed=128, unet=None, base=None):
        base = base or self.tmp / "base"
        if not base.exists():
            base.mkdir()
        mj = self.models_json(base)
        self.patch(T, "MODELS_JSON", mj)
        ref = T.base_binding(base, verify=True)
        tracker = T.PerPromptStatTracker(16, 4)
        tracker.update(["inet-0001", "inet-0001"], [-30.0, -31.0])
        c = T.COMMON
        sampler = T.sampler_doc(c["num_steps"], c["eta"], c["guidance_scale"], c["resolution"],
                                {"_class_name": "DDIMScheduler", "_use_default_values": ["b", "a"]})
        out = self.tmp / "out"
        out.mkdir(exist_ok=True)
        lora_cfg = {"r": 4, "lora_alpha": 4, "target_modules": ["to_k", "to_out.0", "to_q", "to_v"]}
        d = T.save_cut(unet or FakeUnet(), out, epoch, global_step, consumed, ref, sampler, lora_cfg, tracker,
                       {"script_sha256": "sha256:" + "1" * 64, "sampler_code_sha256": T.sampler_code_sha256(),
                        "seed": 3000, "profile": "full"})
        return d, ref, base

    def test_cut_is_what_the_adapter_accepts(self):
        d, ref, base = self.cut()
        ad = A.DdpoSd15Adapter()
        self.assertEqual(d.name, "checkpoint-2")
        self.assertTrue(A.CHECKPOINT_DIR_RE.fullmatch(d.name))
        self.assertIsNone(ad.cut_problem(d, 2))
        self.assertIsNone(ad.model_dir_problem(d))
        self.assertEqual(ad.cut_problem(d, 3), "cut.json epoch is 2, not 3")
        doc = json.loads((d / "cut.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(doc["deployable"]), sorted(R.TRAINER_ROWS["ddpo_sd15"]["components"]))
        self.assertEqual(sorted(doc["files"]), sorted(A.CUT_FILES))
        self.assertEqual((doc["epoch"], doc["global_step"], doc["samples_consumed"], doc["seed"], doc["profile"]),
                         (2, 4, 128, 3000, "full"))
        self.assertEqual(stat.S_IMODE((d / T.LORA_WEIGHTS).stat().st_mode), 0o644)
        self.assertFalse((self.tmp / "out" / ".checkpoint-2.partial").exists())
        # base_ref.json is what the family's EXT-121 check compares with the spec's base directory.
        b = json.loads((d / "base_ref.json").read_text(encoding="utf-8"))
        self.assertEqual(b["files"], ad._base_files(base))
        self.assertEqual(b["files_sha256"], compact_sha256(b["files"]))
        self.assertTrue(A.REVISION_RE.fullmatch(b["revision"]))
        cfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
        self.assertEqual((cfg["base_model_name_or_path"], cfg["base_model_revision"]), (b["repo"], b["revision"]))
        state = json.loads((d / "trainer_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["stat_tracker"], {"inet-0001": [-30.0, -31.0]})

    def test_cut_bytes_do_not_depend_on_the_header_order(self):
        a, _, _ = self.cut(unet=FakeUnet(1))
        shutil.move(str(a), str(self.tmp / "first"))
        b, _, _ = self.cut(unet=FakeUnet(7))
        self.assertEqual((self.tmp / "first" / T.LORA_WEIGHTS).read_bytes(), (b / T.LORA_WEIGHTS).read_bytes())

    def test_existing_cut_is_kept_and_a_stale_partial_replaced(self):
        out = self.tmp / "out"
        (out / ".checkpoint-2.partial").mkdir(parents=True)
        (out / ".checkpoint-2.partial" / "junk").write_text("x", encoding="utf-8")
        d, _, _ = self.cut()
        self.assertIsNone(A.DdpoSd15Adapter().cut_problem(d, 2))
        before = (d / "cut.json").read_bytes()
        again, _, _ = self.cut(global_step=99)
        self.assertEqual(again, d)
        self.assertEqual((d / "cut.json").read_bytes(), before)

    def test_eval_check_cut_accepts_the_cut_and_refuses_changes(self):
        d, _, _ = self.cut()
        self.assertEqual(E.check_cut(d)["epoch"], 2)
        with self.subTest("another sampler code"):
            self.patch(E, "sampler_code_sha256", lambda: "sha256:" + "0" * 64)
            with self.assertRaises(SystemExit):
                E.check_cut(d)
        with self.subTest("a changed file"):
            setattr(E, "sampler_code_sha256", T.sampler_code_sha256)
            (d / "sampler.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                E.check_cut(d)

    def test_eval_check_cut_refuses_another_base(self):
        d, _, _ = self.cut()
        doc = json.loads(E.MODELS_JSON.read_text(encoding="utf-8"))
        doc["models"][T.BASE_NAME]["revision"] = "0" * 40
        E.MODELS_JSON.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaises(SystemExit):
            E.check_cut(d)

    def test_eval_resolve_sampler_takes_the_cuts_sampler(self):
        d, _, _ = self.cut()
        sched = {"_use_default_values": ["a", "b"], "_class_name": "DDIMScheduler"}
        args = E.parse_args(["--checkpoint", str(d), "--out", str(self.tmp / "ev")])
        doc = E.resolve_sampler(args, d, sched)
        self.assertEqual(doc, json.loads((d / "sampler.json").read_text(encoding="utf-8")))
        args = E.parse_args(["--checkpoint", str(d), "--out", str(self.tmp / "ev"), "--num-steps", "50"])
        with self.assertRaises(SystemExit):
            E.resolve_sampler(args, d, sched)
        with self.assertRaises(SystemExit):              # the evaluator's scheduler differs from the cut's
            E.resolve_sampler(E.parse_args(["--checkpoint", str(d), "--out", "x"]), d, {"_class_name": "PNDM"})
        base_only = E.resolve_sampler(E.parse_args(["--base-only", "--out", "x"]), None, sched)
        self.assertEqual(base_only, doc)


# ---------------------------------------------------------------------------
# Run directory, base binding, prompts, env freeze
# ---------------------------------------------------------------------------

class TestRunFiles(Tmp):
    def test_prepare_out_numbers_attempts_and_keeps_earlier_ones(self):
        out = self.tmp / "out"
        self.assertEqual(T.prepare_out(out), 1)
        (out / "samples.jsonl").write_text("{}\n", encoding="utf-8")
        self.assertEqual(T.prepare_out(out), 2)
        self.assertTrue((out / "preempted-attempt-1" / "samples.jsonl").is_file())
        self.assertFalse((out / "samples.jsonl").exists())
        (out / "run_args.json").write_text("{}\n", encoding="utf-8")
        self.assertEqual(T.prepare_out(out), 3)
        self.assertTrue((out / "preempted-attempt-2" / "run_args.json").is_file())
        att = [json.loads(x) for x in (out / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([a["attempt"] for a in att], [1, 2, 3])

    def test_base_binding(self):
        base = self.tmp / "base"
        base.mkdir()
        self.models_json(base)
        ref = T.base_binding(base, verify=True)
        self.assertEqual(ref["files_sha256"], compact_sha256(ref["files"]))
        self.assertEqual(ref["files"], A.DdpoSd15Adapter()._base_files(base))
        self.assertTrue(ref["verified"])
        (base / "model_index.json").write_bytes(b"{} changed\n")
        self.assertFalse(T.base_binding(base, verify=False)["verified"])
        with self.assertRaises(SystemExit):
            T.base_binding(base, verify=True)

    def test_base_binding_refuses_links_extra_files_and_branch_revisions(self):
        base = self.tmp / "base"
        base.mkdir()
        mj = self.models_json(base)
        (base / "extra.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(SystemExit):
            T.base_binding(base, verify=False)
        (base / "extra.txt").unlink()
        shutil.move(str(base / "model_index.json"), str(self.tmp / "model_index.json"))
        os.symlink(self.tmp / "model_index.json", base / "model_index.json")      # the same file set, one a link
        with self.assertRaises(SystemExit):
            T.base_binding(base, verify=True)
        (base / "model_index.json").unlink()
        shutil.move(str(self.tmp / "model_index.json"), str(base / "model_index.json"))
        doc = json.loads(mj.read_text(encoding="utf-8"))
        doc["models"][T.BASE_NAME]["revision"] = "main"
        mj.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaises(SystemExit):
            T.base_binding(base, verify=False)

    def test_read_prompts(self):
        rows, path = T.read_prompts(RUNTIME / "prompts")
        self.assertEqual(len(rows), 248)
        self.assertTrue(all(isinstance(r["prompt"], str) and r["prompt"] for r in rows))
        d = self.tmp / "p"
        d.mkdir()
        (d / "train.jsonl").write_text('{"prompt_id": "a", "prompt": "x"}\n\n{"prompt_id": "a", "prompt": "y"}\n',
                                       encoding="utf-8")
        with self.assertRaises(SystemExit):
            T.read_prompts(d)

    def test_env_freeze_names_the_reference_the_adapter_wants(self):
        out = self.tmp / "out"
        out.mkdir()
        rt = T.write_env_freeze(out)
        lines = (out / "env-freeze.txt").read_text(encoding="utf-8").splitlines()
        self.assertTrue(any(A.REFERENCE_LINE_RE.fullmatch(x) for x in lines))
        self.assertTrue(all("==" in x for x in lines if not x.startswith("#")))
        doc = json.loads((out / "env-runtime.json").read_text(encoding="utf-8"))
        self.assertEqual(doc, rt)
        self.assertEqual(doc["script_sha256"], T.sha256_file(RUNTIME / "train_ddpo.py"))
        self.assertEqual(doc["sampler_code_sha256"], T.sampler_code_sha256())
        self.assertEqual(doc["env_freeze_sha256"], T.sha256_file(out / "env-freeze.txt"))


def main_block_sha256(path: Path, blocks) -> str:
    """SHA-256 of main()'s top-level statements in `blocks`, each (first, last): from the statement matching `first`
    (it must match exactly one) to the next one matching `last`, both included, as ast.unparse writes them (comments
    and layout do not count)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main").body
    parts = []
    for first, last in blocks:
        starts = [k for k, n in enumerate(body) if first(n)]
        if len(starts) != 1:
            raise AssertionError(f"{path.name}: main() has {len(starts)} statements matching the block's first anchor")
        end = next(k for k in range(starts[0], len(body)) if last(body[k]))
        parts.extend(ast.unparse(n) for n in body[starts[0]:end + 1])
    return "sha256:" + hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def assigns(name: str):
    return lambda n: isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)


def epoch_loop(n) -> bool:
    return isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "epoch"


# ---------------------------------------------------------------------------
# A whole run's logs, written with the runtime's functions, through the adapter's consumption check
# ---------------------------------------------------------------------------

class TestRunThroughTheLedger(Tmp):
    """samples.jsonl, train_steps.jsonl, run_args.json settings and the cuts as train_ddpo.py main writes them. The
    per-epoch bookkeeping of main (prompt draw :595-597, sample record :605-625, step grouping :639-682) is followed
    statement by statement, with the runtime's derive_seed, canonical, sha256_bytes, resolve_settings, prepare_out and
    save_cut; the training permutation (a torch.randperm in main) is a stand-in permutation, which the adapter does
    not re-derive."""

    def run_logs(self, profile: str, epochs: int | None = None, attempts: int = 1):
        argv = ["--profile", profile, "--out", str(self.tmp / "out")] + (["--epochs", str(epochs)] if epochs else [])
        args = T.parse_args(argv)
        out = args.out
        for _ in range(attempts):
            attempt = T.prepare_out(out)
        s = T.resolve_settings(args)
        prompts, prompts_file = T.read_prompts(RUNTIME / "prompts")
        base = self.tmp / "base"
        base.mkdir(exist_ok=True)
        self.models_json(base)
        base_ref = T.base_binding(base, verify=True)
        import numpy as np

        sampler = T.sampler_doc(s["num_steps"], s["eta"], s["guidance_scale"], s["resolution"], {})
        tracker = T.PerPromptStatTracker(s["stat_buffer_size"], s["stat_min_count"])
        T.write_json(out / "run_args.json", {"profile": profile, "attempt": attempt, "settings": s, "seed": s["seed"],
                                             "sampler": sampler})
        global_step, samples_consumed, cuts = 0, 0, []
        P, G = s["prompts_per_epoch"], s["images_per_prompt"]
        for epoch in range(1, s["epochs"] + 1):
            rng = np.random.Generator(np.random.PCG64(T.derive_seed("prompts", s["seed"], epoch)))
            chosen = rng.choice(len(prompts), size=P, replace=False)
            plan = [(prompts[int(c)], g) for c in chosen for g in range(G)]
            first_step = global_step + 1
            last_step = global_step + s["updates_per_epoch"] * s["num_inner_epochs"]
            rewards, sample_ids, prompt_ids = [], [], []
            for i, (row, g) in enumerate(plan):
                seed = T.derive_seed("sample", s["seed"], epoch, i)
                kb = 30.0 + (i % 7) * 0.125
                sid = T.sha256_bytes(T.canonical(["ddpo-sd15", s["seed"], attempt, epoch, i, row["prompt_id"], seed,
                                                  g]))
                rewards.append(-kb)
                sample_ids.append(sid)
                prompt_ids.append(row["prompt_id"])
                T.append_jsonl(out / "samples.jsonl", {
                    "epoch": epoch, "sample_index": i, "sample_id": sid, "prompt_id": row["prompt_id"],
                    "prompt": row["prompt"], "member": g, "seed": seed, "jpeg_kb": kb, "reward": -kb,
                    "steps": [first_step, last_step], "attempt": attempt, "utc": T.utc_now()})
            tracker.update(prompt_ids, rewards)
            n = len(plan)
            for inner in range(s["num_inner_epochs"]):
                order = list(np.random.default_rng(epoch).permutation(n))
                batches = [order[i:i + s["train_batch"]] for i in range(0, n, s["train_batch"])]
                consumed = []
                for bi, bidx in enumerate(batches):
                    consumed.extend(sample_ids[int(i)] for i in bidx)
                    if (bi + 1) % s["grad_accum_batches"] == 0:
                        global_step += 1
                        T.append_jsonl(out / "train_steps.jsonl", {"step": global_step, "epoch": epoch, "inner": inner,
                                                                   "sample_ids": consumed, "utc": T.utc_now()})
                        consumed = []
            samples_consumed += n
            if epoch % s["save_every"] == 0 or epoch == s["epochs"]:
                d = T.save_cut(FakeUnet(epoch), out, epoch, global_step, samples_consumed, base_ref, sampler,
                               {"r": 4}, tracker, {"seed": s["seed"], "profile": profile})
                cuts.append((global_step, d))
        return out, cuts, s

    def check(self, profile, **kw):
        out, cuts, s = self.run_logs(profile, **kw)
        ad = A.DdpoSd15Adapter()
        found, incomplete = ad.find_cuts(out)
        self.assertEqual([d for _, d in found], [d for _, d in cuts])
        self.assertEqual(incomplete, [])
        self.assertEqual(ad.consumption_problems(out, RUNTIME / "prompts", found), [])
        plan, bad = ad._ledger_plan(out, RUNTIME / "prompts", found)
        self.assertEqual(bad, [])
        self.assertEqual(len(plan), s["epochs"] * s["samples_per_epoch"] * s["num_inner_epochs"])
        self.assertEqual(int(found[-1][1].name.split("-")[1]), s["epochs"])     # EXT-STAGE: the last epoch is cut
        return out

    def test_cpu_sanity_profile(self):
        self.check("cpu-sanity")

    def test_smoke_profile(self):
        self.check("smoke")

    def test_full_profile_with_the_results_epochs(self):
        out = self.check("full", epochs=12)
        cuts = sorted(p.name for p in out.glob("checkpoint-*"))
        self.assertEqual(cuts, ["checkpoint-12", "checkpoint-4", "checkpoint-8"])

    def test_rerun_after_preemption_records_its_attempt(self):
        self.check("cpu-sanity", attempts=2)

    # main()'s statements that run_logs follows (from `global_step = 0` through the epoch loop). A change
    # there (not to comments or layout) changes this hash: bring run_logs in line with main(), then update it.
    LEDGER_BLOCK_SHA256 = "sha256:45d833544279442711e0f8950e7c0cdae9356122585fe059cc29c263f5d5fb41"

    def test_replica_tracks_mains_bookkeeping(self):
        blocks = [(assigns("global_step"), epoch_loop)]
        self.assertEqual(main_block_sha256(RUNTIME / "train_ddpo.py", blocks), self.LEDGER_BLOCK_SHA256,
                         "main()'s ledger bookkeeping changed: update the replica in TestRunThroughTheLedger and this "
                         "hash")


# ---------------------------------------------------------------------------
# eval_images.py: items, arguments, metric registry
# ---------------------------------------------------------------------------

class TestEvalImages(Tmp):
    def test_make_items_reproduces_the_committed_split(self):
        # A copy of the script in a scratch runtime directory (make_items writes next to the script).
        rt = self.tmp / "rt"
        (rt / "data-source").mkdir(parents=True)
        (rt / "models").mkdir()
        shutil.copyfile(RUNTIME / "eval_images.py", rt / "eval_images.py")
        shutil.copyfile(RUNTIME / "data-source" / "imagenet_classes.txt", rt / "data-source" / "imagenet_classes.txt")
        shutil.copyfile(RUNTIME / "models" / "MODELS.json", rt / "models" / "MODELS.json")
        mod = load_script(rt / "eval_images.py", "_ddpo_runtime_eval_copy")
        args = mod.parse_args(["--make-items", "--classes", str(rt / "data-source" / "imagenet_classes.txt")])
        meta = json.loads((RUNTIME / "items" / "item_set_meta.json").read_text(encoding="utf-8"))
        self.assertEqual((args.n_items, args.split_seed), (meta["split"]["n_items"], meta["split"]["split_seed"]))
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(mod.make_items(args.classes, args.n_items, args.split_seed), 0)
        for rel in ("items/items.jsonl", "prompts/train.jsonl", "constraints/constraint_set.json"):
            self.assertEqual((rt / rel).read_bytes(), (RUNTIME / rel).read_bytes(), rel)
        items = mod.read_items(rt / "items")
        train = [json.loads(x) for x in (rt / "prompts" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual((len(items), len(train)), (150, 248))
        self.assertFalse({r["prompt"] for r in items} & {r["prompt"] for r in train})
        with self.assertRaises(SystemExit):               # never overwrites
            with contextlib.redirect_stdout(io.StringIO()):
                mod.make_items(args.classes, args.n_items, args.split_seed)

    def test_class_names_and_items(self):
        names = E.class_names(RUNTIME / "data-source" / "imagenet_classes.txt")
        self.assertEqual(len(names), E.N_ANIMAL_CLASSES)
        self.assertTrue(all(n and "," not in n and n == n.strip() for n in names))
        d = self.tmp / "items"
        d.mkdir()
        (d / "items.jsonl").write_text('{"item": "a"}\n{"item": "a"}\n', encoding="utf-8")
        with self.assertRaises(SystemExit):
            E.read_items(d)

    def test_parse_args_needs_one_model(self):
        import contextlib
        import io

        for argv in (["--out", "x"], ["--out", "x", "--base-only", "--checkpoint", "c"], ["--base-only"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                E.parse_args(argv)
        a = E.parse_args(["--base-only", "--out", "x"])
        self.assertEqual((a.eval_seeds, a.dtype, a.batch_size), ("11,12", "fp16", 8))
        self.assertEqual([int(x) for x in a.eval_seeds.split(",")],
                         yaml.safe_load((GATE / "registration-gatedtrain.yaml").read_text(encoding="utf-8"))
                         ["clause"]["eval_seeds"])
        # Every option the adapter's argv check knows exists, with the same arity.
        for opt, takes in A.EVAL_OPTIONS.items():
            model = ["--checkpoint", "c"] if opt == "--base-only" else ["--base-only"]
            argv = ["--out", "x", opt] + ([{"--dtype": "fp32"}.get(opt, "1")] if takes else [])
            argv = [opt] if opt == "--make-items" else argv if opt in ("--checkpoint", "--base-only") else model + argv
            self.assertIsNotNone(E.parse_args(argv), opt)
        # ... and the adapter knows every option the script has (an unknown one would pass the argv check unbound).
        import re

        buf = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(buf):
            E.parse_args(["--help"])
        self.assertEqual(set(re.findall(r"(--[a-z][a-z-]*)", buf.getvalue())) - {"--help"}, set(A.EVAL_OPTIONS))

    def test_metrics_are_the_registered_evaluators(self):
        for metric, (ev, direction, rng, _tier, role) in E.METRICS.items():
            if role == "constraint":
                self.assertNotIn(ev, R.EVALUATOR_ROWS)       # reported only (decision 3)
                continue
            row = R.EVALUATOR_ROWS[ev]
            self.assertEqual((row["metric"], row["direction"], list(row["score_range"])), (metric, direction, rng))
            self.assertEqual(tuple(row["error_classes"]), E.ERROR_CLASSES)
        webp = R.EVALUATOR_ROWS["t2i.sd15_webp_items@0.1"]["scorer"]["version"]
        self.assertTrue(webp.startswith(E.SCORER_VERSION + " pillow "))
        self.assertTrue(webp.endswith(json.dumps(E.WEBP, sort_keys=True)))
        jpeg = R.EVALUATOR_ROWS["t2i.sd15_jpeg_items@0.1"]["scorer"]["version"]
        self.assertTrue(jpeg.endswith(f"quality {T.COMMON['jpeg_quality']}"))

    def test_summary_stats(self):
        # stats() is nested in evaluate(); it is taken from the script's source and run on its own (it needs math)
        import math

        tree = ast.parse((RUNTIME / "eval_images.py").read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "stats")
        ns = {"math": math}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(RUNTIME / "eval_images.py"), "exec"), ns)
        stats = ns["stats"]
        # 0.1 + 1e-12 is the same value as 0.1 to 9 decimals: two distinct values among four scores
        got = stats([0.1, 0.2, None, 0.2, 0.1 + 1e-12])
        self.assertEqual((got["n"], got["min"], got["max"], got["distinct"]), (4, 0.1, 0.2, 2))
        self.assertAlmostEqual(got["mean"], 0.15, places=12)
        self.assertAlmostEqual(got["sd"], math.sqrt(4 * 0.05 ** 2 / 3), places=12)     # sample sd, n - 1 = 3
        self.assertEqual(stats([None]), {"n": 0})
        self.assertIsNone(stats([0.5])["sd"])

    def test_link_or_copy(self):
        src = self.tmp / "a.png"
        src.write_bytes(b"png")
        E.link_or_copy(src, self.tmp / "arm" / "images" / "x" / "r0.png")
        self.assertEqual((self.tmp / "arm" / "images" / "x" / "r0.png").read_bytes(), b"png")


if __name__ == "__main__":
    unittest.main()
