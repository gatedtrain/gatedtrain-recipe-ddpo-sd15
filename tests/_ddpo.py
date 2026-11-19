"""Area "image generation, SD1.5" test helpers (verifier-owned; tests/README.md section 11.20, EXT-118 to EXT-130).

Written from the runtime (`runtime/`: README.md "Outputs of `train_ddpo.py`",
"Cut layout", "Seeds", "What gatedtrain needs", "Decisions taken"; `train_ddpo.py` `save_cut`, `base_binding`, `main`;
`eval_images.py` `evaluate`; read 2026-11-12) and its CPU sanity outputs of 2026-11-12, not from gatedtrain. Small real
pieces are copied into `fixtures/ext/ddpo/`: the registered items and the training shard (byte copies of
`area3-sd15/items/items.jsonl` and `prompts/train.jsonl`), a run's `env-freeze.txt` and a cut's `adapter_config.json`
(cpu-sanity run), and the deployment sampler `sampler.json` (20 steps, 512 px; the sampler of the 512-px base
evaluation). Weights are byte stand-ins (minimal safetensors files): nothing in gatedtrain loads them (MM-12).

Layout of `DdpoScenario(root)`:
- `<root>/area3-sd15/` the runtime directory stand-in (mounted read-only at `/inputs/a3`, EXT-112): `train_ddpo.py`,
  `eval_images.py`, `.venv/bin/python` (stand-ins, never executed), `prompts/{train.jsonl, prompts_meta.json}`,
  `items/{items.jsonl, item_set_meta.json}`, `models/stable-diffusion-v1-5/` (the 13 files of the real base with
  stand-in contents) and `models/MODELS.json`;
- `<root>/out/area3-sd15/full-1/`: one training run (seed 3000, 3 epochs of P = 4 prompts x G = 2 images, 2
  optimizer steps per epoch, cuts at epochs 2 and 3) with run_args.json, env-freeze.txt, env-runtime.json,
  samples.jsonl, train_steps.jsonl, train_log.jsonl, memory.jsonl, attempts.jsonl;
- `<root>/eval/{base,cand}/`: two evaluation runs as `eval_images.py` writes them, each with `images/`, `scores.jsonl`
  and the arms `arm-webp_lossless_bpp/` and `arm-jpeg_q95_kb/` (hard-linked images); by default the gatedtrain hashes
  are already filled (as `ext-arm-fill` leaves them);
- spec, registration (gate on `webp_lossless_bpp`, lower is better, superiority, margin 0, sigma 0.6, alpha 0.025,
  N 150, K 2), trainer.yaml, a GPU launch record (gpu_est_h 1.3), a smoke record and the host work's GPU log.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import struct
from pathlib import Path
from typing import Any, Callable

import yaml

EXT = Path(__file__).resolve().parent / "fixtures"     # the recipe's fixtures (gatedtrain: tests/fixtures/ext)
from gatedtrain.testing.ext import ENV_HOST_WORK, ENV_INPUT_ROOTS, ENV_INTERPRETER, ENV_INTERPRETER_TREES, ENV_OUTPUT_ROOT, IMAGE_ID, gpu_step_script, host_config_sha256, launch_label, safetensors_bytes, sha_of, write_canonical
from gatedtrain.ext.arm import META_SCHEMA
from gatedtrain.ext.launch import RECORD_SCHEMA
from gatedtrain.ext.smoke import SMOKE_SCHEMA
from gatedtrain.testing.support import canonical_bytes, file_sha256, manifest_entries, manifest_hash, run_cli

DDPO = EXT / "ddpo"
ADAPTER = "ddpo_sd15"
TRAINER_ID = "rl.ddpo@0.1"
EVALUATOR_ID = "t2i.sd15_webp_items@0.1"
SUPPORT_ID = "t2i.sd15_jpeg_items@0.1"
METRIC = "webp_lossless_bpp"
SUPPORT_METRIC = "jpeg_q95_kb"
AREA3 = Path(__import__("gatedtrain_recipe_ddpo_sd15.rows", fromlist=["RUNTIME"]).RUNTIME)   # the runtime directory
EVAL_IMAGES = AREA3 / "eval_images.py"
EVAL_IMAGES_SHA256 = "sha256:504b374e820f5684e5f5e1c4f9bb0584e18edc9a36d8b12996fcca7e709331c7"
# eval_images.py WEBP and the scorer versions it writes (pillow 13.0.0, libwebp 1.6.0, libjpeg ABI 6.2: the
# 2026-11-12 base evaluation's eval_meta.json and env-runtime.json `pil`).
WEBP = {"format": "WEBP", "lossless": True, "quality": 100, "method": 4, "exact": True}
SCORER = {"name": "webp_lossless_bpp",
          "version": f"eval_images@0.1 pillow 13.0.0 libwebp 1.6.0 {json.dumps(WEBP, sort_keys=True)}",
          "weights_sha256": None, "code_sha256": EVAL_IMAGES_SHA256}
SUPPORT_SCORER = {"name": "jpeg_q95_kb", "version": "eval_images@0.1 pillow 13.0.0 libjpeg 6.2 quality 95",
                  "weights_sha256": None, "code_sha256": EVAL_IMAGES_SHA256}
SAMPLER = json.loads((DDPO / "sampler.json").read_text(encoding="utf-8"))
SAMPLER_KEYS = ("eta", "guidance_scale", "latents_dtype", "negative_prompt", "noise", "num_steps", "resolution",
                "safety_checker", "scheduler", "scheduler_config")
SEEDS = [11, 12]
# train_ddpo.py CUT_FILES, DEPLOYABLE; cut.json is written last.
LORA_WEIGHTS = "pytorch_lora_weights.safetensors"
CUT_FILES = (LORA_WEIGHTS, "adapter_config.json", "sampler.json", "base_ref.json", "trainer_state.json")
DEPLOYABLE = (LORA_WEIGHTS, "adapter_config.json", "sampler.json", "base_ref.json")
# The 13 files of models/stable-diffusion-v1-5 (README "Models"; MODELS.json).
BASE_FILES = ("feature_extractor/preprocessor_config.json", "model_index.json", "scheduler/scheduler_config.json",
              "text_encoder/config.json", "text_encoder/model.fp16.safetensors", "tokenizer/merges.txt",
              "tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
              "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors", "vae/config.json",
              "vae/diffusion_pytorch_model.fp16.safetensors")
BASE_NAME = "stable-diffusion-v1-5"
REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
RUNTIME_CONTAINER = "/inputs/a3"
SEED = 3000
SAMPLER_CODE_SHA = "sha256:7707a4ea20320ef76d97bfa484ed24d8017446a8e7b046017a30c4c1cdf7211c"
SCRIPT_TEXT = "# fixture stand-in for runtime/train_ddpo.py; never executed\n"
EVAL_TEXT = "# fixture stand-in for runtime/eval_images.py; never executed\n"
INTERPRETER_TEXT = "# fixture stand-in for the area-3 venv interpreter; never executed\n"
LABEL = "mm-a3-sd15-full"
ENV_NAMES = ["HF_HUB_OFFLINE", "HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "TOKENIZERS_PARALLELISM",
             "XDG_CACHE_HOME"]
# The fixture profile: train_ddpo.py COMMON plus a small profile (P = 4, G = 2, 2 optimizer steps per epoch).
PROFILE = {"epochs": 3, "prompts_per_epoch": 4, "images_per_prompt": 2, "sample_batch": 4, "train_batch": 2,
           "updates_per_epoch": 2, "save_every": 2}
COMMON = {"num_steps": 20, "eta": 1.0, "guidance_scale": 5.0, "resolution": 512, "learning_rate": 3.0e-4,
          "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_weight_decay": 1.0e-4, "adam_epsilon": 1.0e-8,
          "max_grad_norm": 1.0, "adv_clip_max": 5.0, "clip_range": 1.0e-4, "timestep_fraction": 1.0,
          "train_cfg": True, "num_inner_epochs": 1, "stat_buffer_size": 16, "stat_min_count": 4,
          "reward": "jpeg_compressibility", "jpeg_quality": 95, "precision": "fp16", "gradient_checkpointing": True,
          "allow_tf32": True}


def sha256_text(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def derive_seed(*parts) -> int:
    """train_ddpo.py / eval_images.py `derive_seed`."""
    return int.from_bytes(hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()[:8], "big") >> 1


def compact_hash(obj: Any) -> str:
    """The runtime's own hash encoding (`canonical` in both scripts): compact sorted JSON, not RFC 8785."""
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def write_json(path: Path, doc: Any) -> Path:
    """train_ddpo.py / eval_images.py `write_json`: indent 2, sorted keys, final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, docs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(d, sort_keys=True) + "\n" for d in docs), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Base model, cuts and hashes (README "What gatedtrain needs" item 2; MM-17, MM-18)
# ---------------------------------------------------------------------------

def build_base(d: Path, tag: str = "") -> Path:
    """A plain SD1.5 directory with the real base's 13 file names; weights are minimal safetensors stand-ins."""
    for rel in BASE_FILES:
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".safetensors"):
            p.write_bytes(safetensors_bytes(f"{rel} {tag}"))
        elif rel.endswith(".json"):
            p.write_text(json.dumps({"_fixture": rel, "_class_name": rel.split("/")[0], "tag": tag}, indent=2) + "\n",
                         encoding="utf-8")
        else:
            p.write_text(f"fixture stand-in {rel} {tag}\n", encoding="utf-8")
    return d


def models_json(base: Path) -> dict:
    """models/MODELS.json as fetch_models.py writes it (the base entry only)."""
    return {"schema": "ptc_mm.models", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
            "models": {BASE_NAME: {"repo": REPO, "revision": REVISION,
                                   "files": {rel: {"sha256": file_sha256(base / rel), "size": (base / rel).stat().st_size}
                                             for rel in BASE_FILES}}}}


def base_ref_doc(base: Path, *, revision: str = REVISION) -> dict:
    """train_ddpo.py `base_binding` reduced to the cut's base_ref.json keys."""
    files = {rel: file_sha256(base / rel) for rel in BASE_FILES}
    return {"repo": REPO, "revision": revision, "files": files, "files_sha256": compact_hash(files)}


def lora_bytes(tag: str) -> bytes:
    """`save_lora_adapter` + `canonicalize_safetensors_header`: sorted compact header, space padding to 8 bytes,
    the LoraConfig in `__metadata__.lora_adapter_metadata`."""
    meta = json.dumps({"lora_alpha": 4, "r": 4, "target_modules": ["to_k", "to_out.0", "to_q", "to_v"]}, indent=2,
                      sort_keys=True)
    header = {"__metadata__": {"format": "pt", "lora_adapter_metadata": meta},
              "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight":
                  {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
              "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.weight":
                  {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]}}
    text = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    text += b" " * (-len(text) % 8)
    return struct.pack("<Q", len(text)) + text + hashlib.sha256(tag.encode()).digest()[:16]


def build_cut(out: Path, epoch: int, *, global_step: int, samples_consumed: int, base: Path, script_sha256: str,
              profile: str = "full", seed: int = SEED, tag: str = "", sampler: dict | None = None) -> Path:
    """`checkpoint-<epoch>/` as train_ddpo.py save_cut writes it, `cut.json` last."""
    d = out / f"checkpoint-{epoch}"
    d.mkdir(parents=True)
    (d / LORA_WEIGHTS).write_bytes(lora_bytes(f"lora epoch {epoch} {tag}"))
    shutil.copyfile(DDPO / "adapter_config.json", d / "adapter_config.json")
    write_json(d / "sampler.json", sampler if sampler is not None else SAMPLER)
    write_json(d / "base_ref.json", base_ref_doc(base))
    write_json(d / "trainer_state.json", {"epoch": epoch, "global_step": global_step,
                                          "note": "resume-only information; the run does not resume from a cut "
                                                  "(a preempted run restarts)",
                                          "samples_consumed": samples_consumed, "stat_tracker": {}})
    files = {n: file_sha256(d / n) for n in CUT_FILES}
    write_json(d / "cut.json", {
        "schema": "ptc_mm.ddpo_sd15_cut", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "epoch": epoch, "global_step": global_step, "samples_consumed": samples_consumed, "files": files,
        "deployable": list(DEPLOYABLE), "created_at": "2026-11-12T20:00:00Z", "script_sha256": script_sha256,
        "sampler_code_sha256": SAMPLER_CODE_SHA, "seed": seed, "profile": profile})
    return d


def adapter_hash(cut: Path) -> str:
    """Manifest hash (T0-09) of the deployable components of a cut (cut.json `deployable`)."""
    entries = [e for e in manifest_entries(cut) if e["path"] in DEPLOYABLE]
    return "sha256:" + hashlib.sha256(canonical_bytes(entries)).hexdigest()


def lora_model_hash(cut: Path, base: Path) -> str:
    """MM-18: sha256(canonical_json({base: base_hash, adapter: adapter_hash}))."""
    return sha_of({"base": manifest_hash(base), "adapter": adapter_hash(cut)})


# ---------------------------------------------------------------------------
# A training run (train_ddpo.py main: samples.jsonl at sampling time, train_steps.jsonl before each step)
# ---------------------------------------------------------------------------

def settings(profile: dict | None = None, seed: int = SEED) -> dict:
    s = {**COMMON, **(profile or PROFILE)}
    s["seed"] = seed
    s["samples_per_epoch"] = s["prompts_per_epoch"] * s["images_per_prompt"]
    s["grad_accum_batches"] = s["samples_per_epoch"] // (s["train_batch"] * s["updates_per_epoch"])
    s["num_train_timesteps"] = max(1, int(s["num_steps"] * s["timestep_fraction"]))
    return s


def simulate(prompts: list[dict], s: dict, *, attempt: int = 1) -> tuple[list[dict], list[dict], dict[int, tuple]]:
    """(samples.jsonl lines, train_steps.jsonl lines, {epoch: (global_step, samples_consumed)}) of a run."""
    seed, P, G = s["seed"], s["prompts_per_epoch"], s["images_per_prompt"]
    samples, steps, state, gs, consumed = [], [], {}, 0, 0
    for e in range(1, s["epochs"] + 1):
        chosen = random.Random(derive_seed("prompts", seed, e)).sample(range(len(prompts)), P)
        plan = [(prompts[c], g) for c in chosen for g in range(G)]
        first, last = gs + 1, gs + s["updates_per_epoch"] * s["num_inner_epochs"]
        ids = []
        for i, (row, g) in enumerate(plan):
            noise = derive_seed("sample", seed, e, i)
            sid = sha256_text(canonical_bytes(["ddpo-sd15", seed, attempt, e, i, row["prompt_id"], noise, g]))
            kb = round(40.0 + (i * 7 % 13) - 0.4 * e, 3)
            ids.append(sid)
            samples.append({"attempt": attempt, "epoch": e, "jpeg_kb": kb, "member": g, "prompt": row["prompt"],
                            "prompt_id": row["prompt_id"], "reward": -kb, "sample_id": sid, "sample_index": i,
                            "seed": noise, "steps": [first, last], "utc": "2026-11-12T19:00:00Z"})
        per = len(plan) // s["updates_per_epoch"]
        for inner in range(s["num_inner_epochs"]):                 # train_ddpo.py:639-682
            order = list(range(len(plan)))
            random.Random(derive_seed("perm", seed, e, inner)).shuffle(order)
            for k in range(s["updates_per_epoch"]):
                gs += 1
                steps.append({"epoch": e, "inner": inner, "sample_ids": [ids[j] for j in order[k * per:(k + 1) * per]],
                              "step": gs, "utc": "2026-11-12T19:00:05Z"})
        consumed += len(plan)
        state[e] = (gs, consumed)
    return samples, steps, state


def run_args(*, s: dict, profile: str, attempt: int, base_container: str, train_container: str,
             train_sha256: str, files_sha256: str, script_sha256: str) -> dict:
    """train_ddpo.py run_args.json (the keys ext-collect reads, as the cpu-sanity run wrote them)."""
    return {
        "schema": "ptc_mm.ddpo_sd15_run_args", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "profile": profile, "attempt": attempt, "settings": s,
        "lora": json.loads((DDPO / "adapter_config.json").read_text(encoding="utf-8")),
        "seed": s["seed"], "seed_derivation": "derive_seed(*parts) = first 8 bytes of SHA-256(':'.join(parts)) >> 1",
        "base_model": base_container,
        "base_ref": {"repo": REPO, "revision": REVISION, "files_sha256": files_sha256, "verified": True},
        "train_file": train_container, "train_file_sha256": train_sha256, "n_train_prompts": 248,
        "sampler": SAMPLER, "sampler_sha256": compact_hash(SAMPLER),
        "consumption_log": "samples.jsonl", "step_log": "train_steps.jsonl",
        "device": "cuda", "dtype": "torch.float16", "gpu": "NVIDIA GeForce RTX 4090", "torch": "2.15.0+cu132",
        "torch_cuda": "13.2", "sampler_code_sha256": SAMPLER_CODE_SHA, "script_sha256": script_sha256,
        "created_at": "2026-11-12T19:00:00Z",
    }


def build_run(out: Path, *, runtime: Path, base: Path, script_sha256: str, profile: str = "full",
              s: dict | None = None) -> tuple[Path, dict[int, tuple]]:
    s = s or settings()
    out.mkdir(parents=True)
    prompts = read_jsonl(runtime / "prompts" / "train.jsonl")
    samples, steps, state = simulate(prompts, s)
    shutil.copyfile(DDPO / "env-freeze.txt", out / "env-freeze.txt")
    write_json(out / "env-runtime.json", {"python": "3.12.16", "executable": f"{RUNTIME_CONTAINER}/.venv/bin/python",
                                          "image_cuda_version": "13.0.3", "script_sha256": script_sha256,
                                          "sampler_code_sha256": SAMPLER_CODE_SHA,
                                          "env_freeze_sha256": file_sha256(out / "env-freeze.txt")})
    write_json(out / "run_args.json", run_args(
        s=s, profile=profile, attempt=1, base_container=f"{RUNTIME_CONTAINER}/models/{BASE_NAME}",
        train_container=f"{RUNTIME_CONTAINER}/prompts/train.jsonl",
        train_sha256=file_sha256(runtime / "prompts" / "train.jsonl"),
        files_sha256=base_ref_doc(base)["files_sha256"], script_sha256=script_sha256))
    write_jsonl(out / "samples.jsonl", samples)
    write_jsonl(out / "train_steps.jsonl", steps)
    write_jsonl(out / "train_log.jsonl", [{"step": st["step"], "epoch": st["epoch"], "loss": 0.0} for st in steps])
    write_jsonl(out / "memory.jsonl", [{"epoch": e, "seconds_epoch": 180.0} for e in state])
    write_jsonl(out / "attempts.jsonl", [{"attempt": 1, "started_at": "2026-11-12T19:00:00Z"}])
    for e, (gs, consumed) in state.items():
        if e % s["save_every"] == 0 or e == s["epochs"]:
            build_cut(out, e, global_step=gs, samples_consumed=consumed, base=base, script_sha256=script_sha256,
                      profile=profile, seed=s["seed"])
    return out, state


# ---------------------------------------------------------------------------
# Evaluation runs (eval_images.py evaluate: images, scores.jsonl, one arm per metric)
# ---------------------------------------------------------------------------

def baseline_score(i: int, r: int) -> float:
    return round(8.0 + 0.37 * (i % 9) + 0.11 * r, 6)


def candidate_score(i: int, r: int) -> float:
    return round(baseline_score(i, r) - 0.45 - 0.05 * ((i + r) % 3), 6)


def build_eval(out: Path, items: list[dict], score_fn: Callable[[int, int], float], *, tag: str,
               model: str | None, base_model: str | None, item_set: str | None, filled: bool = True,
               metrics: tuple[str, ...] = (METRIC, SUPPORT_METRIC)) -> Path:
    """`<out>/images/<item>/r<r>.png`, `<out>/scores.jsonl` and `<out>/arm-<metric>/` per metric with hard-linked
    images, as eval_images.py writes them. `filled`: the gatedtrain hashes are set (model, item set, RFC 8785 sampler
    hash), as after ext-arm-fill; otherwise the evaluator's own output (nulls, compact-JSON sampler hash)."""
    out.mkdir(parents=True)
    rows = []
    for r, seed in enumerate(SEEDS):
        for i, it in enumerate(items):
            rel = f"images/{it['item']}/r{r}.png"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_bytes(b"\x89PNG\r\n\x1a\n" + f"stand-in {tag} {it['item']} {r}".encode())
            webp = score_fn(i, r)
            rows.append({"item": it["item"], "repeat": r, "seed": seed, "output_ref": {"path": rel,
                         "sha256": file_sha256(out / rel)}, "webp_lossless_bpp": webp,
                         "png_bpp": round(webp * 1.4, 6), "jpeg_q95_kb": round(webp * 10.5, 3), "aesthetic": 5.2,
                         "siglip_rank": 1, "prompt_violation": 0, "nonfinite": False, "errors": {}})
    write_jsonl(out / "scores.jsonl", rows)
    scorers = {METRIC: (EVALUATOR_ID, SCORER), SUPPORT_METRIC: (SUPPORT_ID, SUPPORT_SCORER)}
    for metric in metrics:
        arm = out / f"arm-{metric}"
        arm.mkdir()
        lines = []
        for row in rows:
            dst = arm / row["output_ref"]["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(out / row["output_ref"]["path"], dst)
            sub = {k: row[k] for k in ("webp_lossless_bpp", "png_bpp", "jpeg_q95_kb", "aesthetic", "siglip_rank",
                                       "prompt_violation")}
            sub["nonfinite"] = 0
            lines.append({"item": row["item"], "repeat": row["repeat"], "seed": row["seed"], "score": row[metric],
                          "error": None, "sub_metrics": sub, "output_ref": row["output_ref"]})
        write_jsonl(arm / "items.jsonl", lines)
        evaluator, scorer = scorers[metric]
        write_json(arm / "eval_meta.json", {
            "schema": META_SCHEMA, "schema_version": "1.0", "evaluator_id": evaluator, "adapter": ADAPTER,
            "metric": metric, "direction": "lower_is_better", "score_range": [0, None],
            "checkpoint": {"model_hash": model if filled else None, "base_model_hash": base_model if filled else None},
            "normalizer_sha256": None, "env_version_sha256": None,
            "item_set_hash": item_set if filled else None, "eval_seeds": list(SEEDS),
            "sampler": SAMPLER, "sampler_hash": sha_of(SAMPLER) if filled else compact_hash(SAMPLER),
            "scorer": dict(scorer), "framework": {"name": "diffusers", "version": "0.42.1"},
            "created_at": "2026-11-12T21:00:00Z"})
    return out


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def smoke_record(*, base_model: str, item_set: str, **overrides) -> dict[str, Any]:
    """A PASS smoke.json (MM-38) judged on the deciding metric only (decision 5 in runtime/README.md "Decisions taken"): 20
    baseline webp_lossless_bpp scores, range [0, null] (no mean rule), not binary."""
    rec = {"schema": SMOKE_SCHEMA, "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
           "adapter": ADAPTER, "area": 2, "base_model_hash": base_model, "item_set_hash": item_set,
           "peak_gpu_mem_gb": 10.4, "host_ram_peak_gb": 11.0, "swap_peak_gb": 0.0, "seconds_per_step": 90.0,
           "projected_gpu_h": 1.3, "gpu_log_total_h": 4.43,
           "baseline_scores": [baseline_score(i, 0) for i in range(20)], "error_count": 0, "verdict": "PASS",
           "reasons": []}
    rec.update(overrides)
    return rec


def guard_log(host_work: Path, label: str, *, lines: list[str] | None = None) -> Path:
    """gpu-step.sh's log: one finished attempt of 3600 s."""
    host_work.mkdir(parents=True, exist_ok=True)
    lines = lines if lines is not None else [f"2026-11-11T09:00:00Z START {label} attempt=1",
                                             f"2026-11-11T10:00:00Z END {label} attempt=1 rc=0"]
    p = host_work / "gpu-guard.log"
    p.write_text("".join(l + "\n" for l in lines), encoding="utf-8")
    return p


class DdpoScenario:
    """A whole area-3 run collected against the SD1.5 base (module docstring)."""

    def __init__(self, root: Path, *, n_items: int = 150, reg_mutate: Callable[[dict], None] | None = None,
                 overlap: int = 0, filled: bool = True, inner_epochs: int = 1):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.runtime = root / "area3-sd15"
        self.data = self.runtime / "prompts"
        self.items_dir = self.runtime / "items"
        self.data.mkdir(parents=True)
        self.items_dir.mkdir(parents=True)
        all_items = read_jsonl(DDPO / "items.jsonl")[:n_items]
        write_jsonl(self.items_dir / "items.jsonl", all_items)
        write_json(self.items_dir / "item_set_meta.json", {"schema": "ptc_mm.sd15_item_set", "n_items": len(all_items)})
        prompts = read_jsonl(DDPO / "train.jsonl")
        for row, item in zip(prompts[:overlap], all_items):          # exact overlap: a shard prompt equals an item's
            row["prompt"] = item["prompt"]
        if overlap:
            write_jsonl(self.data / "train.jsonl", prompts)
        else:
            shutil.copyfile(DDPO / "train.jsonl", self.data / "train.jsonl")
        write_json(self.data / "prompts_meta.json", {"schema": "ptc_mm.sd15_train_prompts", "n_train": len(prompts)})
        self.item_rows = all_items
        self.items = [r["item"] for r in all_items]
        self.base = build_base(self.runtime / "models" / BASE_NAME)
        write_json(self.runtime / "models" / "MODELS.json", models_json(self.base))
        self.script = self.runtime / "train_ddpo.py"
        self.script.write_text(SCRIPT_TEXT, encoding="utf-8")
        (self.runtime / "eval_images.py").write_text(EVAL_TEXT, encoding="utf-8")
        (self.runtime / "uv.lock").write_text("# lock stand-in\n", encoding="utf-8")
        self.interpreter = self.runtime / ".venv" / "bin" / "python"
        self.interpreter.parent.mkdir(parents=True)
        self.interpreter.write_text(INTERPRETER_TEXT, encoding="utf-8")
        (root / "trainer.yaml").write_text("# area 3 trainer config (fixture)\nbackend: ddpo_sd15\nprofile: full\n",
                                           encoding="utf-8")
        self.host_work = root / "host-work"
        self.output_root = root / "out"
        gpu_step_script(self.host_work)
        (self.host_work / "gpu-log.md").write_bytes((EXT / "gpu-log.fixture.md").read_bytes())
        self.stage_out, self.state = build_run(self.output_root / "area3-sd15" / "full-1", runtime=self.runtime,
                                               base=self.base, script_sha256=file_sha256(self.script),
                                               s={**settings(), "num_inner_epochs": inner_epochs})
        self.cuts = sorted(self.stage_out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        self.final_cut = self.cuts[-1]
        self.base_model = manifest_hash(self.base)
        self.cand_model = lora_model_hash(self.final_cut, self.base)
        self.item_set_hash = manifest_hash(self.items_dir)
        self.reg_mutate = reg_mutate
        self.render()
        self.base_eval = build_eval(root / "eval" / "base", self.item_rows, baseline_score, tag="base",
                                    model=self.base_model, base_model=self.base_model, item_set=self.item_set_hash,
                                    filled=filled)
        self.cand_eval = build_eval(root / "eval" / "cand", self.item_rows, candidate_score, tag="cand",
                                    model=self.cand_model, base_model=self.base_model, item_set=self.item_set_hash,
                                    filled=filled)
        self.baseline = self.base_eval / f"arm-{METRIC}"
        self.candidate = self.cand_eval / f"arm-{METRIC}"
        self.launch = write_canonical(root / "launch.json", self.launch_record())
        self.label = self.launch_record()["label"]
        self.smoke = write_canonical(root / "smoke.json", smoke_record(base_model=self.base_model,
                                                                         item_set=self.item_set_hash))
        guard_log(self.host_work, self.label)

    def env(self) -> dict[str, str]:
        return {ENV_HOST_WORK: str(self.host_work), ENV_OUTPUT_ROOT: str(self.output_root),
                ENV_INPUT_ROOTS: str(self.root), ENV_INTERPRETER: f"{RUNTIME_CONTAINER}/.venv/bin/python",
                ENV_INTERPRETER_TREES: str(self.interpreter.parent)}

    @property
    def paths(self) -> dict[str, Path]:
        return {"spec": self.root / "spec.yaml", "registration": self.root / "registration.yaml"}

    def render(self, reg_mutate: Callable[[dict], None] | None = None,
               spec_mutate: Callable[[dict], None] | None = None) -> None:
        text = (DDPO / "registration.template.yaml").read_text(encoding="utf-8")
        text = text.replace("{item_set_hash}", self.item_set_hash).replace("{sampler}", json.dumps(SAMPLER))
        text = text.replace("{evaluator_hash}", sha_of({"evaluator": EVALUATOR_ID, "metric": METRIC,
                                                         "direction": "lower_is_better", "scorer": SCORER}))
        mutate = reg_mutate or self.reg_mutate
        if mutate is not None:
            reg = yaml.safe_load(text)
            mutate(reg)
            text = yaml.safe_dump(reg, sort_keys=False)
        (self.root / "registration.yaml").write_text(text, encoding="utf-8")
        spec = (DDPO / "spec.template.yaml").read_text(encoding="utf-8")
        spec = spec.replace("{registration_sha256}", file_sha256(self.root / "registration.yaml")[len("sha256:"):])
        spec = spec.replace("{base}", str(self.base)).replace("{data}", str(self.data))
        spec = spec.replace("{items}", str(self.items_dir))
        if spec_mutate is not None:
            doc = yaml.safe_load(spec)
            spec_mutate(doc)
            spec = yaml.safe_dump(doc, sort_keys=False)
        (self.root / "spec.yaml").write_text(spec, encoding="utf-8")

    def launch_record(self, **overrides) -> dict[str, Any]:
        """ext-launch's record of the full run (README "Request shape for `ext-launch`", full; gpus true)."""
        rec = {"schema": RECORD_SCHEMA, "schema_version": "2.0", "tags": ["prototype", "unpromotable"],
               "adapter": ADAPTER, "area": 2, "profile": "full", "label": LABEL, "host_work": str(self.host_work),
               "runtime": {"kind": "venv_in_container", "image": IMAGE_ID,
                           "venv_lock_sha256": file_sha256(self.runtime / "uv.lock")},
               "framework": {"name": "diffusers", "version": "0.42.1", "commit": None},
               "training_seed": SEED, "gpus": True,
               "argv": [f"{RUNTIME_CONTAINER}/.venv/bin/python", f"{RUNTIME_CONTAINER}/train_ddpo.py", "--profile",
                        "full", "--out", "/out", "--seed", str(SEED), "--epochs", "3"],
               "mounts": [{"host": str(self.runtime), "container": RUNTIME_CONTAINER, "mode": "ro"},
                          {"host": str(self.stage_out), "container": "/out", "mode": "rw"}],
               "tmpfs": ["/tmp"], "network": "none", "network_reason": None, "user": "1000:1000",
               "env_names": list(ENV_NAMES), "gpu_est_h": 1.3, "gpu_used_h": 4.43,
               "output_dir": str(self.stage_out), "output_root": str(self.output_root),
               "gpu_step_sha256": file_sha256(self.host_work / "bin" / "gpu-step.sh"), "allow_network": False,
               "interpreter_path": f"{RUNTIME_CONTAINER}/.venv/bin/python",
               "interpreter_sha256": file_sha256(self.interpreter),
               "script_path": f"{RUNTIME_CONTAINER}/train_ddpo.py", "script_sha256": file_sha256(self.script),
               "host_config_sha256": host_config_sha256(self.host_work, self.output_root, [self.root],
                                                         interpreter=f"{RUNTIME_CONTAINER}/.venv/bin/python",
                                                         interpreter_trees=[self.interpreter.parent])}
        rec.update(overrides)
        rec["label"] = launch_label(rec)
        return rec

    def rewrite_launch(self, **overrides) -> None:
        rec = self.launch_record(**overrides)
        self.launch.write_bytes(canonical_bytes(rec))
        self.label = rec["label"]
        guard_log(self.host_work, self.label)

    def edit_json(self, path: Path, fn: Callable[[dict], None]) -> None:
        doc = json.loads(path.read_text(encoding="utf-8"))
        fn(doc)
        write_json(path, doc)

    def edit_jsonl(self, path: Path, fn: Callable[[list[dict]], list[dict]]) -> None:
        write_jsonl(path, fn(read_jsonl(path)))

    def reseal(self, cut: Path) -> None:
        """Rewrite cut.json's file hashes after a cut file was edited (the cut stays complete)."""
        self.edit_json(cut / "cut.json", lambda d: d.update(files={n: file_sha256(cut / n) for n in CUT_FILES}))

    def collect(self, runs_root: Path, *, baseline: Path | None = None, candidate: Path | None = None,
                launch: Path | None = None, smoke: Path | None = None, timeout: int = 300):
        args = ["ext-collect", str(self.paths["spec"]), "--adapter", ADAPTER, "--stage-out", str(self.stage_out),
                "--launch-record", str(launch or self.launch), "--smoke-record", str(smoke or self.smoke),
                "--eval-baseline", str(baseline or self.baseline), "--eval-candidate", str(candidate or self.candidate),
                "--runs-root", str(runs_root)]
        return run_cli(args, cwd=self.root, timeout=timeout, env=self.env())

    def fill(self, arm: Path, checkpoint: Path, *, base: Path | None = None, adapter_name: str = ADAPTER):
        args = ["ext-arm-fill", "--adapter", adapter_name, "--checkpoint", str(checkpoint), "--items",
                str(self.items_dir)]
        if base is not None:
            args += ["--base", str(base)]
        return run_cli(args + [str(arm)], cwd=self.root, env=self.env())
