"""DDPO SD1.5 + LoRA adapter (area "image generation, SD1.5": the DDPO port `train_ddpo.py` on diffusers 0.42.1 +
peft 0.21.2; MULTIMODAL-SPEC.md section 6 row 2a and runtime/README.md "Decisions taken"; tests/README.md
section 11.20, EXT-118 to EXT-130).

MM-12: a pure-Python class that imports no framework, torch or pickle loader. It reads JSON, JSONL, the text
`env-freeze.txt` and safetensors headers (standard library `struct` and `json`), and hashes everything else.

Run layout, read from the runtime's code `runtime/train_ddpo.py` (and its
README.md "Outputs of `train_ddpo.py`", "Cut layout", "Seeds", "What gatedtrain needs"; read 2026-11-12; where the two
differ the code is followed, docs/findings.md):

- `save_cut` (train_ddpo.py:442-474) writes `checkpoint-<E>/` (E the 1-based epoch) into
  `.checkpoint-<E>.partial/` and renames it: `pytorch_lora_weights.safetensors` (header canonicalized, :414-439),
  `adapter_config.json`, `sampler.json` (the deployment sampler), `base_ref.json` (repo, revision, per-file SHA-256 of
  the base, `files_sha256`), `trainer_state.json` (resume only), then `cut.json` last: `epoch`, `global_step`,
  `samples_consumed`, `files` (SHA-256 of the five), `deployable` (`DEPLOYABLE`, :108: the LoRA weights, adapter
  config, sampler and base reference), `script_sha256`, `sampler_code_sha256`, `seed`, `profile`. Cuts at every
  `save_every` epochs and at the last epoch (:695);
- `samples.jsonl` (:622-625): one line per generated sample at sampling time, `{epoch, sample_index, sample_id,
  prompt_id, prompt, member, seed, jpeg_kb, reward, steps: [first, last], attempt, utc}`; `train_steps.jsonl`
  (:671-672): one line per optimizer step, before the step, `{step, epoch, inner, sample_ids, utc}`. The samples of
  epoch e are consumed by the steps of epoch e only; with `num_inner_epochs` I each sample is in one step per inner
  epoch. Nothing is filtered (no zero-advantage drop), so outcome `filtered` never occurs (MM-28);
- `run_args.json` (:558-584): `profile`, `attempt`, `settings` (`epochs`, `samples_per_epoch`, `updates_per_epoch`,
  `num_inner_epochs`, ...), `seed`, `base_model` (the container path of the base directory), `base_ref` (`repo`,
  `revision`, `files_sha256`, `verified`), `train_file` and `train_file_sha256`, `sampler`, `sampler_code_sha256`,
  `script_sha256`; `env-runtime.json` (:328-351) `script_sha256` (EXT-83) and `sampler_code_sha256`;
  `env-freeze.txt` `name==version` lines plus `# ddpo algorithm reference kvablack/ddpo-pytorch <40 hex> (ported)`.

Identity (MM-17, MM-18): the plain SD1.5 base directory's model hash is its manifest hash (the baseline arm evaluates
the base alone); a cut's adapter hash is the manifest hash of its deployable files, and its model hash is
sha256(canonical_json({base: base manifest hash, adapter: adapter hash})).
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

from gatedtrain.canonical import (canonical_json, entries_hash, file_sha256, manifest_entries, manifest_hash, sha256_bytes,
                          text_lines)
from gatedtrain.errors import Problem
from gatedtrain.ext.registry import EVALUATOR_ROWS, TRAINER_ROWS
from gatedtrain.ext.sdk import (compact_sha256, derive_seed, is_int as _int, plain_file as _plain_file, read_cut_marker,
                   read_json as _read_json, read_jsonl, read_run_args, runtime_argv_problems, runtime_script_problems,
                   safetensors_problem, sha_or_none as _sha_or_none, single_shard as _shard)
from gatedtrain.families.diffusers_lora import DiffusersLoraAdapter

NAME = "ddpo_sd15"
LIBRARY = "the DDPO port train_ddpo.py on diffusers + peft (cuts by its save_cut, cut.json written last)"
CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")        # train_ddpo.py save_cut; `.checkpoint-<E>.partial` never
CUT_MARKER = "cut.json"                                       # written last (train_ddpo.py save_cut)
LORA_WEIGHTS = "pytorch_lora_weights.safetensors"
CUT_FILES = (LORA_WEIGHTS, "adapter_config.json", "sampler.json", "base_ref.json", "trainer_state.json")
COMPONENTS = tuple(TRAINER_ROWS[NAME]["components"])          # train_ddpo.py DEPLOYABLE (EXT-120)
BASE_REF = "base_ref.json"
FRAMEWORK_PACKAGE = "diffusers"
PINS = tuple(TRAINER_ROWS[NAME]["pins"])
REFERENCE_LINE_RE = re.compile(r"^# ddpo algorithm reference kvablack/ddpo-pytorch [0-9a-f]{40} \(ported\)$")
FREEZE_FILE = "env-freeze.txt"
RUNTIME_FILE = "env-runtime.json"
SAMPLES = "samples.jsonl"
STEPS = "train_steps.jsonl"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
# The plain SD1.5 base directory (README "Models": model_index.json, scheduler, tokenizer, feature-extractor config,
# fp16 safetensors of UNet, VAE and text encoder with their configs).
BASE_REQUIRED = ("model_index.json", "scheduler/scheduler_config.json", "unet/config.json", "vae/config.json",
                 "text_encoder/config.json", "tokenizer/tokenizer_config.json")
BASE_WEIGHT_DIRS = ("unet", "vae", "text_encoder")
MAX_SEED = 2 ** 63                                            # derive_seed: 8 bytes >> 1

CHECKPOINT_COMPLETENESS = {
    "library": LIBRARY,
    "rule": ("a checkpoint-<E> directory is a cut iff its cut.json, written last after an atomic rename from "
             ".checkpoint-<E>.partial, parses with epoch = E, an integer global_step and samples_consumed, a deployable "
             "list of the LoRA weights, adapter_config.json, sampler.json and base_ref.json, and files listing exactly "
             "the other files of the directory (those four and trainer_state.json), each with its SHA-256; the LoRA "
             "safetensors file is whole; .partial directories are never cuts; the cut's last step included is its "
             "global_step"),
    "complete_dir_pattern": CHECKPOINT_DIR_RE.pattern,
    "sources": ["runtime/train_ddpo.py:442-474 save_cut (cut.json last, os.replace)",
                "runtime/train_ddpo.py:105-108 CUT_FILES, DEPLOYABLE",
                "runtime/README.md \"Cut layout\""],
    "verified": True,                                # the layout was read from the CPU sanity run's cut (README)
}
SEED_DERIVATION = ("train_ddpo.py derive_seed(*parts) = first 8 bytes of SHA-256(':'.join(parts)) >> 1: epoch e's "
                   "prompts from numpy PCG64(derive_seed('prompts', seed, e)), sample i's noise from a CPU "
                   "torch.Generator seeded with derive_seed('sample', seed, e, i) (recorded per sample in "
                   "samples.jsonl and in the ledger's seed_inputs), the training order from derive_seed('perm', seed, "
                   "e, inner), the LoRA init from derive_seed('torch', seed). The training seed is run_args.json `seed`, "
                   "which must equal the launch record's training_seed and every cut's cut.json `seed`; 05 section 14 "
                   "H(run_seed, step, data_ref, sample_index) is not claimed.")


EVAL_SCRIPT = "eval_images.py"
TRAIN_SCRIPT = "train_ddpo.py"
DEFAULT_BASE_REL = "models/stable-diffusion-v1-5"            # eval_images.py DEFAULT_BASE = HERE / this
# eval_images.py parse_args (:636-656): option -> takes a value (EXT-156: --items, --limit; EXT-133: --base).
EVAL_OPTIONS = {"--make-items": False, "--classes": True, "--n-items": True, "--split-seed": True, "--checkpoint": True,
                "--base-only": False, "--base": True, "--items": True, "--out": True, "--eval-seeds": True,
                "--model-hash": True, "--base-model-hash": True, "--item-set-hash": True, "--limit": True,
                "--batch-size": True, "--device": True, "--dtype": True, "--no-aesthetic": False, "--num-steps": True,
                "--eta": True, "--guidance-scale": True, "--resolution": True}


class DdpoSd15Adapter(DiffusersLoraAdapter):
    """The MM-12 interface for DDPO + LoRA runs on SD1.5 (EXT-118 to EXT-130)."""

    # Layout constants, read through self so a family base class reads the subclass's values.
    BASE_REF = BASE_REF
    CHECKPOINT_DIR_RE = CHECKPOINT_DIR_RE
    COMPONENTS = COMPONENTS
    CUT_FILES = CUT_FILES
    CUT_MARKER = CUT_MARKER
    DEFAULT_BASE_REL = DEFAULT_BASE_REL
    EVAL_OPTIONS = EVAL_OPTIONS
    EVAL_SCRIPT = EVAL_SCRIPT
    FRAMEWORK_PACKAGE = FRAMEWORK_PACKAGE
    FREEZE_FILE = FREEZE_FILE
    LORA_WEIGHTS = LORA_WEIGHTS
    MAX_SEED = MAX_SEED
    NAME = NAME
    PINS = PINS
    REFERENCE_LINE_RE = REFERENCE_LINE_RE
    REVISION_RE = REVISION_RE
    RUNTIME_FILE = RUNTIME_FILE
    SAMPLES = SAMPLES
    STEPS = STEPS
    TRAIN_SCRIPT = TRAIN_SCRIPT

    name = NAME
    trainer_id = TRAINER_ROWS[NAME]["uses"][0]
    evaluator_ids = tuple(TRAINER_ROWS[NAME]["evaluators"])
    framework_name = FRAMEWORK_PACKAGE
    checkpoint_completeness = CHECKPOINT_COMPLETENESS
    seed_derivation = SEED_DERIVATION
    binds_spec = bool(TRAINER_ROWS[NAME]["spec_binding"])
    # ext-collect messages (EXT-123, EXT-124).
    consumption_fix = "collect the run's own samples.jsonl and train_steps.jsonl as the run wrote them (MM-24, MM-27)"
    payload_source = "a training prompt's (the training shard or samples.jsonl; exact overlap, MM-11, EXT-124)"


    # --- the base directory and cuts (MM-17, MM-18, MM-20; EXT-119) ---


    @staticmethod
    def base_dir_problem(d: Path) -> str | None:
        """None when `d` is a plain SD1.5 directory: the required configs and whole safetensors weights of the UNet,
        VAE and text encoder (README "Models")."""
        for rel in BASE_REQUIRED:
            if not _plain_file(Path(d) / rel):
                return f"{rel} is missing (not an SD1.5 directory)"
        for sub in BASE_WEIGHT_DIRS:
            weights = sorted(Path(d, sub).glob("*.safetensors"))
            if not weights:
                return f"{sub}/ has no safetensors weights"
            for w in weights:
                bad = safetensors_problem(w)
                if bad:
                    return f"{sub}/{bad}"
        return None


    def _cut_files_problem_unguarded(self, d: Path) -> str | None:
        doc, why = read_cut_marker(d)
        if doc is None:
            return why
        files = doc["files"]
        if sorted(files) != sorted(self.CUT_FILES):
            return f"{self.CUT_MARKER} lists {sorted(files)}, not {sorted(self.CUT_FILES)}"
        present = sorted(p.name for p in Path(d).iterdir())
        if present != sorted([*self.CUT_FILES, self.CUT_MARKER]):
            extra = sorted(set(present) - set(self.CUT_FILES) - {self.CUT_MARKER})
            return f"the directory holds {extra or present}, which {self.CUT_MARKER} does not list"
        for name in self.CUT_FILES:
            if not _plain_file(Path(d) / name):
                return f"{name} is not a plain file"
            if file_sha256(Path(d) / name) != files[name]:
                return f"{name} is not the SHA-256 {self.CUT_MARKER} records"
        if not (isinstance(doc.get("deployable"), list) and all(isinstance(x, str) for x in doc["deployable"])
                and sorted(doc["deployable"]) == sorted(self.COMPONENTS)):          # review 11 L1
            return f"{self.CUT_MARKER} deployable is {doc.get('deployable')!r}, not {list(self.COMPONENTS)}"
        for key in ("epoch", "global_step", "samples_consumed"):
            if not (_int(doc.get(key)) and doc[key] >= 1):
                return f"{self.CUT_MARKER} {key} is {doc.get(key)!r}, not a positive integer"
        return safetensors_problem(Path(d) / self.LORA_WEIGHTS)


    def _base_entries(self, base: Path) -> list[dict]:
        """The base directory's manifest entries, walked once per adapter instance (2.1 GB on the real base)."""
        key = str(Path(base).resolve())
        if key not in self._base_cache:
            self._base_cache[key] = manifest_entries(Path(base))
        return self._base_cache[key]


    # --- consumption (MM-24, MM-27; EXT-123) ---

    def _ledger_plan(self, stage_out: Path, data: Path, cuts: list) -> tuple[list[tuple[dict, int]], list[str]]:
        """([(samples.jsonl line, optimizer step)] for the samples of the epochs up to the last cut, in file order, one
        pair per step that applied the sample) and the problems that refuse the log (empty when it is consistent)."""
        args, why = read_run_args(stage_out)
        if args is None:
            return [], [why]
        if not cuts:
            return [], ["no complete cut"]
        s = args.get("settings") if isinstance(args.get("settings"), dict) else {}
        spe, upe, inner_n = s.get("samples_per_epoch"), s.get("updates_per_epoch"), s.get("num_inner_epochs")
        group, run_seed = s.get("images_per_prompt"), args.get("seed")
        if not all(_int(x) and x > 0 for x in (spe, upe, inner_n, group)) or not _int(run_seed):
            return [], ["run_args.json has no positive integer settings.samples_per_epoch, updates_per_epoch, "
                        "num_inner_epochs, images_per_prompt and integer seed"]
        shard = _shard(data)
        if shard is None:
            return [], [f"the spec's training prompts {data} hold {len(list(Path(data).glob('*.jsonl')))} shards, not "
                        "the one shard the run read"]
        rows, why = read_jsonl(Path(data) / shard)
        if rows is None:
            return [], [f"the training shard: {why}"]
        prompt_of = {r.get("prompt_id"): r.get("prompt") for r in rows}
        marks = []
        for _, d in cuts:
            doc, why = read_cut_marker(d)
            if doc is None:
                return [], [f"{Path(d).name}: {why}"]
            marks.append((doc.get("epoch"), doc.get("global_step"), doc.get("samples_consumed"), Path(d).name))
        last_epoch = max(m[0] for m in marks)
        samples, why = read_jsonl(Path(stage_out) / self.SAMPLES)
        if samples is None:
            return [], [why]
        steps, why = read_jsonl(Path(stage_out) / self.STEPS)
        if steps is None:
            return [], [why]
        out: list[str] = []

        def bad(msg: str) -> bool:
            out.append(msg)
            return len(out) >= 8

        # samples.jsonl lines (train_ddpo.py:622-625)
        by_epoch: dict[int, list[dict]] = {}
        seen_ids: set[str] = set()
        for n, x in enumerate(samples, start=1):
            e, i, sid, pid, seed, member = (x.get(k) for k in ("epoch", "sample_index", "sample_id", "prompt_id",
                                                                "seed", "member"))
            if not (_int(e) and e >= 1 and _int(i) and i >= 0 and isinstance(sid, str) and isinstance(pid, str)
                    and _int(member) and member >= 0):
                if bad(f"{self.SAMPLES} line {n} has no integer epoch, sample_index, member and string sample_id, prompt_id"):
                    return [], out
                continue
            if not (_int(seed) and 0 <= seed < self.MAX_SEED):
                if bad(f"{self.SAMPLES} line {n}: seed {seed!r} is not a 63-bit integer noise seed"):
                    return [], out
                continue
            reward, kb = x.get("reward"), x.get("jpeg_kb")
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (reward, kb)):
                if bad(f"{self.SAMPLES} line {n}: reward {reward!r} or jpeg_kb {kb!r} is not a finite number"):
                    return [], out
                continue
            # EXT-132 (review M2): the fields train_ddpo.py derives (:605, :615-625) are re-derived, so the ledger's
            # seed inputs and sample keys reproduce the sample (MM-29).
            if seed != derive_seed("sample", run_seed, e, i):
                bad(f"{self.SAMPLES} line {n}: seed {seed} is not derive_seed('sample', {run_seed}, {e}, {i})")
            if member != i % group:
                bad(f"{self.SAMPLES} line {n}: member {member} is not sample_index {i} % images_per_prompt {group}")
            # reward = -kb exactly: train_ddpo.py appends -kb, a float negation (exact), and json writes both with
            # repr, which round-trips; no tolerance is needed.
            if reward != -kb:
                bad(f"{self.SAMPLES} line {n}: reward {reward!r} is not -jpeg_kb ({-kb!r})")
            want_sid = compact_sha256(["ddpo-sd15", run_seed, x.get("attempt"), e, i, pid, seed, member])
            if sid != want_sid:
                bad(f"{self.SAMPLES} line {n}: sample_id is not the hash train_ddpo.py derives ({want_sid[:19]}...)")
            if len(out) >= 8:
                return [], out
            if x.get("attempt") != args.get("attempt"):
                if bad(f"{self.SAMPLES} line {n}: attempt {x.get('attempt')!r} is not run_args.json attempt "
                       f"{args.get('attempt')!r}"):
                    return [], out
            if pid not in prompt_of:
                if bad(f"{self.SAMPLES} line {n}: prompt_id {pid} is not a row of the training shard {shard}"):
                    return [], out
            elif x.get("prompt") != prompt_of[pid]:
                if bad(f"{self.SAMPLES} line {n}: prompt {x.get('prompt')!r} is not the shard's prompt of {pid}"):
                    return [], out
            if sid in seen_ids:
                if bad(f"{self.SAMPLES} line {n}: sample_id repeats"):
                    return [], out
            seen_ids.add(sid)
            by_epoch.setdefault(e, []).append(x)
        if out:
            return [], out
        for e in range(1, last_epoch + 1):
            got = by_epoch.get(e, [])
            if sorted(x["sample_index"] for x in got) != list(range(spe)):
                if bad(f"{self.SAMPLES} epoch {e} logs {len(got)} samples (sample_index "
                       f"{sorted(x['sample_index'] for x in got)[:6]}...), not 0 .. {spe - 1} (samples_per_epoch)"):
                    return [], out
        # train_steps.jsonl lines (train_ddpo.py:671-672)
        nums = [st.get("step") for st in steps]
        if not all(_int(k) for k in nums) or nums != list(range(1, len(steps) + 1)):     # review 10 L3: true == 1
            return [], out + [f"{self.STEPS} steps {nums[:10]}{' ...' if len(nums) > 10 else ''} are not consecutive from 1"]
        epoch_of = {x["sample_id"]: e for e, xs in by_epoch.items() for x in xs}
        applied: dict[str, list[int]] = {}
        per_epoch: dict[int, list[dict]] = {}
        for st in steps:
            e, inner, ids = st.get("epoch"), st.get("inner"), st.get("sample_ids")
            if not (_int(e) and _int(inner) and 0 <= inner < inner_n and isinstance(ids, list)
                    and all(isinstance(i, str) for i in ids)):
                return [], out + [f"{self.STEPS} step {st['step']} has no integer epoch, inner < num_inner_epochs and list of "
                                  "sample_ids"]
            per_epoch.setdefault(e, []).append(st)
            if e > last_epoch:
                continue
            for sid in ids:
                if epoch_of.get(sid) != e:
                    if bad(f"{self.STEPS} step {st['step']} (epoch {e}) applies sample {sid[:19]}..., which is "
                           + (f"a sample of epoch {epoch_of[sid]}" if sid in epoch_of else f"not in {self.SAMPLES}")):
                        return [], out
                    continue
                applied.setdefault(sid, []).append(st["step"])
        for e in range(1, last_epoch + 1):
            sts = per_epoch.get(e, [])
            if len(sts) != upe * inner_n:
                if bad(f"{self.STEPS} epoch {e} has {len(sts)} optimizer steps, not updates_per_epoch x num_inner_epochs "
                       f"= {upe * inner_n}"):
                    return [], out
            for inner in range(inner_n):
                ids = [sid for st in sts if st["inner"] == inner for sid in st["sample_ids"]]
                want = sorted(x["sample_id"] for x in by_epoch.get(e, []))
                if sorted(ids) != want:
                    if bad(f"{self.STEPS} epoch {e} inner {inner}: the steps apply {len(ids)} sample ids, not each of the "
                           f"epoch's {len(want)} samples exactly once (a sample in no step or in two)"):
                        return [], out
        # EXT-158 (review 9 L2, the ddpo analogue): train_ddpo.py stops after settings.epochs, so a step or sample of a
        # later epoch is another run's or an edited log.
        epochs = s.get("epochs")
        beyond = sorted({e for e in per_epoch if not (_int(epochs) and e <= epochs)}
                        | {e for e in by_epoch if not (_int(epochs) and e <= epochs)})
        if beyond:
            bad(f"{self.STEPS} or {self.SAMPLES} log epochs {beyond[:6]} beyond run_args.json settings.epochs {epochs!r}")
        if out:
            return [], out
        # steps of an epoch follow the previous epoch's (epochs in order) and samples name their steps' range
        order = [st.get("epoch") for st in steps]
        if order != sorted(order):
            return [], [f"{self.STEPS} epochs are not in order"]
        for x in (x for e, xs in by_epoch.items() if e <= last_epoch for x in xs):
            nums_e = [st["step"] for st in per_epoch.get(x["epoch"], [])]
            want = [min(nums_e), max(nums_e)] if nums_e else None
            if x.get("steps") != want:                    # EXT-132: [first_step, last_step] of the epoch (:598-599)
                if bad(f"{self.SAMPLES} sample {x['sample_id'][:19]}... (epoch {x['epoch']}) names steps "
                       f"{x.get('steps')!r}, not the epoch's steps {want}"):
                    return [], out
        # MM-27: each cut's counts equal the log up to it
        for epoch, gs, consumed, name in marks:
            want_gs = max((st["step"] for e, sts in per_epoch.items() if e <= epoch for st in sts), default=0)
            if gs != want_gs:
                bad(f"{name} cut.json global_step is {gs!r}, but {self.STEPS} logs {want_gs} steps up to epoch {epoch}")
            upto = gs if _int(gs) else 0
            want_n = len({sid for sid, nums in applied.items() if any(t <= upto for t in nums)})
            if consumed != want_n:
                bad(f"{name} cut.json samples_consumed is {consumed!r}, but the steps up to {gs!r} applied {want_n} "
                    "samples")
        if out:
            return [], out
        plan = [(x, step) for x in samples if x["epoch"] <= last_epoch for step in applied.get(x["sample_id"], [])]
        return plan, []


    def consumption(self, stage_out: Path, data: Path, cuts: list, *, run_id: str = "",
                    attempt_id: str = "0") -> dict[str, Any]:
        """Precision `row`, derivation `observed` (MM-24): one ledger record per (generated sample, optimizer step
        that applied it) of the epochs up to the last cut, from samples.jsonl joined with train_steps.jsonl.
        `data_ref.row` = sha256(prompt_id), shard = the spec's one prompt shard, `seed_inputs.seed` = the sample's
        noise seed, `rewards` = [its JPEG reward], `sampler_config_hash` = the RFC 8785 hash of run_args.json `sampler`,
        `sample_id` = H(run, attempt, epoch, step, prompt_id, seed, member) (NUL-joined text)."""
        from gatedtrain.ledger import WRITER_ID, check_precision

        plan, bad = self._ledger_plan(stage_out, data, cuts)
        if bad:                                          # collect refuses these before writing
            raise ValueError("; ".join(bad))
        check_precision("row", "observed")
        args, _ = read_run_args(stage_out)
        sampler = args.get("sampler")
        sampler_hash = sha256_bytes(canonical_json(sampler)) if isinstance(sampler, dict) else None
        shard = _shard(data)
        snapshot = manifest_hash(data)
        rows = []
        for x, step in plan:
            key = "\x00".join(str(v) for v in (run_id, attempt_id, x["epoch"], step, x["prompt_id"], x["seed"],
                                               x["member"]))
            rows.append({
                "run_id": run_id, "attempt_id": attempt_id, "step": step,
                "sample_id": "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest(),
                "data_ref": {"snapshot": snapshot, "shard": shard,
                             "row": sha256_bytes(x["prompt_id"].encode("utf-8")), "mixture": None, "component": None},
                "parent_ref": None, "outcome": "consumed", "reason": None, "rule_version": None, "gen_version": None,
                "reward_epoch": None, "rewards": [float(x["reward"])], "sampler_config_hash": sampler_hash,
                "seed_inputs": {"seed": x["seed"]}, "chunk": None, "precision": "row", "derivation": "observed",
                "writer_id": WRITER_ID,
            })
        last = max(step for step, _ in cuts)
        return {"precision": "row", "rows": rows, "derivation": "observed",
                "reason": (f"{self.SAMPLES} logs every generated sample at sampling time and {self.STEPS} the samples each "
                           f"optimizer step applied, before the step (MM-24, MM-27): {len(rows)} records of steps 1 .. "
                           f"{last}, data_ref.row = sha256(prompt_id) in shard {shard}, derivation observed; no sample "
                           "is filtered (no zero-advantage drop), so no `filtered` record exists")}

    # --- exact overlap (MM-11; EXT-124) ---


    # --- pins and checks (MM-21, MM-29, MM-33; EXT-121, EXT-122, EXT-125) ---

    def framework_identity(self, stage_out: Path, launch: dict) -> dict[str, Any]:
        """`diffusers==<v>` in env-freeze.txt equals the launch record's framework; the freeze pins peft, torch and
        transformers and names the algorithm reference commit (MM-33). Raises ValueError with the reason."""
        freeze = Path(stage_out) / self.FREEZE_FILE
        if not _plain_file(freeze):
            raise ValueError(f"{freeze} is missing")
        lines = [l.strip() for l in text_lines(freeze.read_text(encoding="utf-8"))]
        versions: dict[str, str] = {}
        for line in lines:
            if "==" in line and not line.startswith("#"):
                name, _, version = line.partition("==")
                versions[name.strip().lower().replace("_", "-")] = version.strip()
        missing = [p for p in self.PINS if p not in versions]
        if missing:
            raise ValueError(f"{freeze} has no `<name>==<version>` line for {', '.join(missing)} (pins of {self.NAME})")
        if not any(self.REFERENCE_LINE_RE.fullmatch(l) for l in lines):
            raise ValueError(f"{freeze} has no `# ddpo algorithm reference kvablack/ddpo-pytorch <40 hex> (ported)` "
                             "line (the algorithm reference the port names)")
        version = versions[self.FRAMEWORK_PACKAGE]
        fw = launch.get("framework") if isinstance(launch.get("framework"), dict) else {}
        if fw.get("name") != self.FRAMEWORK_PACKAGE or fw.get("version") != version:
            raise ValueError(f"{freeze} pins {self.FRAMEWORK_PACKAGE}=={version}, but the launch record claims "
                             f"framework {fw.get('name')!r} {fw.get('version')!r}")
        return {"name": self.FRAMEWORK_PACKAGE, "version": version, "freeze_sha256": file_sha256(freeze)}


    BASE_FIX = ("train from the spec's SD1.5 base directory inside a read-only mount (train_ddpo.py --base, default "
                "<area3-sd15>/models/stable-diffusion-v1-5, re-hashed at start), and record that mount in the launch "
                "record")
    DATA_FIX = ("train on the spec's prompt shard inside a read-only mount (train_ddpo.py --prompts, recorded in "
                "run_args.json train_file and train_file_sha256)")

    def argv_problems(self, argv: Any, *, profile: str | None = None) -> list[str]:
        """EXT-133 (review 7 L1), EXT-156 (review 9 L1), ext-launch and --recheck: argv[1] is train_ddpo.py or
        eval_images.py of the runtime directory; eval_images.py loads `--base` and `--items` without hashing them and
        records neither in its outputs, so an evaluation may use only the defaults inside the read-only runtime mount;
        `--limit` is refused under `--profile full`. A training run's base is bound at collect."""
        return runtime_argv_problems(argv, profile=profile, options_of={self.TRAIN_SCRIPT: {}, self.EVAL_SCRIPT: self.EVAL_OPTIONS},
                                     eval_script=self.EVAL_SCRIPT, default_base_rel=self.DEFAULT_BASE_REL,
                                     rule="EXT-133, EXT-156")[0]


