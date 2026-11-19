#!/usr/bin/env python
"""DDPO (clipped PPO on the DDIM denoising chain) with a LoRA adapter on Stable Diffusion 1.5, prototype area
"image generation, SD1.5" (MULTIMODAL-SPEC section 6 row 2a; prototype, unpromotable).

A minimal port of kvablack/ddpo-pytorch at commit 1958463f020112c9a7bc85768d296daacc2e1b4b (MIT License,
Copyright (c) 2023 Kevin Black) onto current diffusers + peft. Ported pieces, cited by file:line of that commit:
  - `ddim_step_with_logprob`: ddpo_pytorch/diffusers_patch/ddim_with_logprob.py:39-192 (epsilon prediction);
  - the sampling loop with classifier-free guidance: ddpo_pytorch/diffusers_patch/pipeline_with_logprob.py:181-220;
  - the PPO update over every denoising step: scripts/train.py:455-593;
  - `PerPromptStatTracker`: ddpo_pytorch/stat_tracking.py:5-30 (`min_count` configurable);
  - the reward `jpeg_compressibility`: ddpo_pytorch/rewards.py:7-29 (JPEG quality 95, size in kB, negated);
  - hyperparameters: config/base.py:5-111 and config/dgx.py:8-38 (`compressibility`).
Why a port and not the repository itself: README.md "Trainer choice".

Changes against ddpo-pytorch, all launcher side (README.md "Changes against ddpo-pytorch"): LoRA through peft
(`UNet2DConditionModel.add_adapter`) instead of the removed `LoRAAttnProcessor(hidden_size, ...)`; noise drawn from a
recorded per-sample seed (CPU generator, then moved to the device) instead of the global RNG; prompts drawn per epoch
from a registered training shard, P prompts x G images, instead of `random.choice` per sample; an observed
consumption log; cuts written atomically with a completion marker; no wandb.

Outputs in --out (the run's only read-write mount):
  env-freeze.txt     `name==version` lines of the runtime, written before torch is imported (MM-33)
  env-runtime.json   interpreter, image CUDA version, lock and script SHA-256 (EXT-83), sampler code SHA-256
  run_args.json      every setting, seeds and their derivation, base model binding (repo, revision, file hashes),
                     data binding (train_file and its SHA-256)
  samples.jsonl      observed consumption log (MM-24, MM-27): one line per generated sample, written at sampling time
  train_steps.jsonl  one line per optimizer step: the sample ids whose gradients it applied, written before the step
  train_log.jsonl    per optimizer step: loss, approx_kl, clipfrac, grad norm, scaler scale
  memory.jsonl       per epoch: seconds (sampling, training), peak device memory, peak RSS, reward statistics
  attempts.jsonl     one line per start and finish of this command (gpu-step.sh reruns after a preemption)
  checkpoint-<E>/    the cut after epoch E (1-based); `cut.json` written last

Usage (inside the container; README.md has the request shape):
  python train_ddpo.py --profile smoke|full --out /out [--base DIR] [--prompts DIR] [--seed S]
  python train_ddpo.py --profile cpu-sanity --device cpu --out DIR      # host CPU check, tiny settings
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "models" / "stable-diffusion-v1-5"
DEFAULT_PROMPTS = HERE / "prompts"
MODELS_JSON = HERE / "models" / "MODELS.json"
BASE_NAME = "stable-diffusion-v1-5"

# LoRA (TRL's DDPO port of the same recipe used peft r=4, alpha=4, gaussian init on the attention projections;
# ddpo-pytorch's LoRAAttnProcessor covered the same four projections of every attention block with rank 4).
LORA = {"r": 4, "lora_alpha": 4, "init_lora_weights": "gaussian",
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"]}

COMMON = {
    # sampler (the deployment sampler too; eval_images.py checks the cut's sampler.json against its own)
    "num_steps": 20,              # ddpo-pytorch: 50 (config/base.py:45); 20 to fit the GPU budget (README)
    "eta": 1.0,                   # config/base.py:48
    "guidance_scale": 5.0,        # config/base.py:50
    "resolution": 512,
    # training (config/base.py:58-90, config/dgx.py:19-24)
    "learning_rate": 3.0e-4,
    "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_weight_decay": 1.0e-4, "adam_epsilon": 1.0e-8,
    "max_grad_norm": 1.0,
    "adv_clip_max": 5.0,
    "clip_range": 1.0e-4,
    "timestep_fraction": 1.0,
    "train_cfg": True,
    "num_inner_epochs": 1,
    # per-prompt statistics (stat_tracking.py); min_count 4 = one epoch's group of G images of the prompt
    "stat_buffer_size": 16,
    "stat_min_count": 4,
    # reward (rewards.py:7-29)
    "reward": "jpeg_compressibility",
    "jpeg_quality": 95,
    # memory knobs
    "precision": "fp16",          # frozen UNet, VAE and text encoder in fp16, LoRA parameters fp32, GradScaler
    "gradient_checkpointing": True,
    "allow_tf32": True,
}
PROFILES = {
    # epochs, prompts per epoch P, images per prompt G, sample batch, train batch, optimizer steps per epoch, cut
    # every N epochs. 64 samples per epoch, 2 optimizer steps of 32 samples (config/dgx.py: 256 samples, 4 steps).
    "smoke": {"epochs": 2, "prompts_per_epoch": 16, "images_per_prompt": 4, "sample_batch": 8, "train_batch": 2,
              "updates_per_epoch": 2, "save_every": 1},
    # full: 24 epochs = 1,536 samples, 48 optimizer steps, cuts 4, 8, ..., 24; about 1.2 GPU-h at the estimated
    # 3 min per epoch (README.md "Profiles and cost"); set --epochs from the smoke's measured seconds per epoch.
    "full": {"epochs": 24, "prompts_per_epoch": 16, "images_per_prompt": 4, "sample_batch": 8, "train_batch": 2,
             "updates_per_epoch": 2, "save_every": 4},
    # host CPU sanity check only (README.md "CPU sanity check"): fp32, 256 px, 2 steps, 2 prompts x 2 images.
    "cpu-sanity": {"epochs": 1, "prompts_per_epoch": 2, "images_per_prompt": 2, "sample_batch": 2, "train_batch": 1,
                   "updates_per_epoch": 2, "save_every": 1, "num_steps": 2, "resolution": 256, "precision": "fp32",
                   "gradient_checkpointing": False},
}
CUT_MARKER = "cut.json"
LORA_WEIGHTS = "pytorch_lora_weights.safetensors"
CUT_FILES = (LORA_WEIGHTS, "adapter_config.json", "sampler.json", "base_ref.json", "trainer_state.json")
DEPLOYABLE = (LORA_WEIGHTS, "adapter_config.json", "sampler.json", "base_ref.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def canonical(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def write_json(path: Path, doc) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# BEGIN SHARED SAMPLER (byte-identical in train_ddpo.py and eval_images.py; its SHA-256 is `sampler_code_sha256`)
# ---------------------------------------------------------------------------

def derive_seed(*parts) -> int:
    """63-bit seed from the SHA-256 of the parts joined by ':' (recorded derivation, MM-29)."""
    return int.from_bytes(hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()[:8], "big") >> 1


def load_sd15(base_dir, device, dtype):
    """SD1.5 components from the plain base directory (fp16 safetensors variant), no safety checker."""
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    base_dir = str(base_dir)
    kw = {"variant": "fp16", "torch_dtype": dtype, "local_files_only": True}
    unet = UNet2DConditionModel.from_pretrained(base_dir, subfolder="unet", **kw).to(device)
    vae = AutoencoderKL.from_pretrained(base_dir, subfolder="vae", **kw).to(device)
    text_encoder = CLIPTextModel.from_pretrained(base_dir, subfolder="text_encoder", **kw).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(base_dir, subfolder="tokenizer", local_files_only=True)
    # ddpo-pytorch switches the pipeline's PNDM scheduler to DDIM with the same config (scripts/train.py:110).
    scheduler = DDIMScheduler.from_pretrained(base_dir, subfolder="scheduler", local_files_only=True)
    for m in (unet, vae, text_encoder):
        m.requires_grad_(False)
    return {"unet": unet, "vae": vae, "text_encoder": text_encoder, "tokenizer": tokenizer, "scheduler": scheduler}


def encode_prompts(sd, prompts, device):
    import torch

    ids = sd["tokenizer"](list(prompts), return_tensors="pt", padding="max_length", truncation=True,
                          max_length=sd["tokenizer"].model_max_length).input_ids.to(device)
    with torch.no_grad():
        return sd["text_encoder"](ids)[0]


def ddim_step_with_logprob(scheduler, model_output, timestep, sample, eta, prev_sample=None, noise=None):
    """ddpo-pytorch ddim_with_logprob.py:39-192 for epsilon prediction without thresholding or clipping (the SD1.5
    scheduler config: clip_sample false). `timestep` is a (B,) long tensor; float32 throughout. Returns
    (prev_sample, log_prob) with log_prob the mean over the latent dimensions (:183-190)."""
    import torch

    def bcast(t):
        return t.reshape(t.shape + (1,) * (sample.ndim - t.ndim)).to(sample.device, torch.float32)

    model_output = model_output.float()
    sample = sample.float()
    t_cpu = timestep.detach().cpu().long()
    prev_t = t_cpu - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_t = torch.clamp(prev_t, 0, scheduler.config.num_train_timesteps - 1)   # :97 (as in the source)
    acp = scheduler.alphas_cumprod.float()
    alpha_prod_t = acp.gather(0, t_cpu)
    alpha_prod_t_prev = torch.where(prev_t >= 0, acp.gather(0, prev_t), scheduler.final_alpha_cumprod.float())
    beta_prod_t = 1 - alpha_prod_t
    variance = ((1 - alpha_prod_t_prev) / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)   # :22-36
    std_dev_t = bcast(eta * variance ** 0.5)
    a_t, a_prev, b_t = bcast(alpha_prod_t), bcast(alpha_prod_t_prev), bcast(beta_prod_t)
    pred_original_sample = (sample - b_t ** 0.5 * model_output) / a_t ** 0.5                   # :116-119
    pred_sample_direction = (1 - a_prev - std_dev_t ** 2) ** 0.5 * model_output                # :159-161
    prev_sample_mean = a_prev ** 0.5 * pred_original_sample + pred_sample_direction            # :164-166
    if prev_sample is None:
        prev_sample = prev_sample_mean + std_dev_t * noise                                     # :174-181
    prev_sample = prev_sample.float()
    log_prob = (-((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std_dev_t ** 2)
                - torch.log(std_dev_t) - math.log(math.sqrt(2 * math.pi)))                     # :184-188
    return prev_sample, log_prob.mean(dim=tuple(range(1, log_prob.ndim)))                     # :190


def predict_noise(unet, latents, timesteps, embeds, neg_embeds, guidance_scale):
    """Classifier-free guidance as pipeline_with_logprob.py:187-206 and scripts/train.py:494-521: one UNet call on
    [uncond, cond]; float32 out."""
    import torch

    dtype = unet.conv_in.weight.dtype             # the frozen base dtype (LoRA parameters are float32)
    lat = torch.cat([latents, latents]).to(dtype)
    t = torch.cat([timesteps, timesteps])
    emb = torch.cat([neg_embeds, embeds]).to(dtype)
    out = unet(lat, t, encoder_hidden_states=emb, return_dict=False)[0].float()
    uncond, cond = out.chunk(2)
    return uncond + guidance_scale * (cond - uncond)


def sample_with_logprob(sd, unet, embeds, neg_embeds, seeds, num_steps, eta, guidance_scale, resolution, device):
    """One batch: initial latents and every step's noise from a CPU torch.Generator seeded with the sample's seed
    (device independent), DDIM with eta, CFG. Returns images in [0, 1] (B, 3, H, W) float32 on the CPU, the latents
    (B, num_steps + 1, 4, h, w) and log-probs (B, num_steps) on the device, and the timesteps (num_steps,)."""
    import torch

    sched = sd["scheduler"]
    sched.set_timesteps(num_steps, device="cpu")
    timesteps = sched.timesteps.clone()
    h = resolution // 8
    gens = [torch.Generator("cpu").manual_seed(int(s)) for s in seeds]
    shape = (4, h, h)
    latents = torch.stack([torch.randn(shape, generator=g, dtype=torch.float32) for g in gens]).to(device)
    latents = latents * sched.init_noise_sigma
    all_latents, all_logp = [latents], []
    with torch.no_grad():
        for t in timesteps:
            tb = torch.full((latents.shape[0],), int(t), dtype=torch.long, device=device)
            eps = predict_noise(unet, latents, tb, embeds, neg_embeds, guidance_scale)
            noise = torch.stack([torch.randn(shape, generator=g, dtype=torch.float32) for g in gens]).to(device)
            latents, logp = ddim_step_with_logprob(sched, eps, tb, latents, eta, noise=noise)
            all_latents.append(latents)
            all_logp.append(logp)
        vae = sd["vae"]
        img = vae.decode((latents / vae.config.scaling_factor).to(next(vae.parameters()).dtype),
                         return_dict=False)[0].float()
        img = (img / 2 + 0.5).clamp(0, 1).cpu()
    return img, torch.stack(all_latents, dim=1), torch.stack(all_logp, dim=1), timesteps


def to_uint8(img):
    """(3, H, W) float in [0, 1] -> (H, W, 3) uint8, as ddpo-pytorch rewards.py:10-11."""
    import torch

    return (img * 255).round().clamp(0, 255).to(torch.uint8).numpy().transpose(1, 2, 0).copy()


def jpeg_kb(arr, quality=95) -> float:
    """ddpo-pytorch rewards.py:7-17: PIL JPEG at quality 95, size in kB (bytes / 1000)."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.tell() / 1000


def sampler_doc(num_steps, eta, guidance_scale, resolution, scheduler_config) -> dict:
    """The deployment sampler (08 section 3.6 'scheduler config'), stored in every cut as sampler.json. diffusers
    builds `_use_default_values` from a set, so its order varies between processes: sorted here."""
    scheduler_config = {k: (sorted(v) if k == "_use_default_values" else v) for k, v in scheduler_config.items()}
    return {"scheduler": "DDIMScheduler", "scheduler_config": scheduler_config, "num_steps": num_steps, "eta": eta,
            "guidance_scale": guidance_scale, "resolution": resolution, "negative_prompt": "",
            "noise": "per sample: torch.Generator('cpu').manual_seed(seed); initial latents then one draw per step",
            "latents_dtype": "float32", "safety_checker": None}

# ---------------------------------------------------------------------------
# END SHARED SAMPLER
# ---------------------------------------------------------------------------


def sampler_code_sha256() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    start = text.index("# BEGIN SHARED SAMPLER")
    end = text.index("# END SHARED SAMPLER")
    return sha256_bytes(text[start:end].encode("utf-8"))


class PerPromptStatTracker:
    """ddpo-pytorch stat_tracking.py:5-30 with `min_count` configurable."""

    def __init__(self, buffer_size: int, min_count: int):
        from collections import deque

        self._deque = deque
        self.buffer_size, self.min_count, self.stats = buffer_size, min_count, {}

    def update(self, prompts, rewards):
        import numpy as np

        prompts, rewards = np.array(prompts), np.array(rewards, dtype=np.float64)
        adv = np.empty_like(rewards)
        for p in np.unique(prompts):
            pr = rewards[prompts == p]
            buf = self.stats.setdefault(p, self._deque(maxlen=self.buffer_size))
            buf.extend(pr)
            if len(buf) < self.min_count:
                mean, std = np.mean(rewards), np.std(rewards) + 1e-6
            else:
                mean, std = np.mean(buf), np.std(buf) + 1e-6
            adv[prompts == p] = (pr - mean) / std
        return adv


# ---------------------------------------------------------------------------
# Pre-step (MM-33)
# ---------------------------------------------------------------------------

def write_env_freeze(out: Path) -> dict:
    pkgs = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            pkgs[name.lower().replace("_", "-")] = dist.version
    lines = [f"{k}=={pkgs[k]}" for k in sorted(pkgs)]
    lines.append("# ddpo algorithm reference kvablack/ddpo-pytorch 1958463f020112c9a7bc85768d296daacc2e1b4b (ported)")
    (out / "env-freeze.txt").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    lock = HERE / "uv.lock"
    runtime = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "image_cuda_version": os.environ.get("CUDA_VERSION"),
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "sampler_code_sha256": sampler_code_sha256(),
        "uid": os.getuid(),
        "gid": os.getgid(),
        "cpu_count": os.cpu_count(),
        "env_freeze_sha256": sha256_file(out / "env-freeze.txt"),
    }
    write_json(out / "env-runtime.json", runtime)
    return runtime


def prepare_out(out: Path) -> int:
    """As area2-drones train_ppo.py: a non-empty --out is a rerun after a preemption; earlier contents move to
    `preempted-attempt-<k>/` (never deleted). Returns the attempt number."""
    out.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out.iterdir() if not p.name.startswith("preempted-attempt-")]
    previous = sorted(p for p in out.iterdir() if p.name.startswith("preempted-attempt-"))
    attempt = len(previous) + 1
    if existing:
        attempt = len(previous) + 2
        dest = out / f"preempted-attempt-{attempt - 1}"
        dest.mkdir()
        for p in existing:
            p.rename(dest / p.name)
        if (dest / "attempts.jsonl").exists():
            shutil.copyfile(dest / "attempts.jsonl", out / "attempts.jsonl")
    append_jsonl(out / "attempts.jsonl", {"attempt": attempt, "started_at": utc_now(), "argv": sys.argv})
    return attempt


# ---------------------------------------------------------------------------
# Bindings
# ---------------------------------------------------------------------------

def base_binding(base: Path, verify: bool) -> dict:
    """Base model reference (MM-18, MM-21): repo, 40-hex revision, and the SHA-256 of every file, from
    models/MODELS.json; with `verify`, every file is re-hashed and must match (about 2.1 GB)."""
    record = json.loads(MODELS_JSON.read_text(encoding="utf-8"))["models"][BASE_NAME]
    files = record["files"]
    for p in sorted(base.rglob("*")):
        if p.is_symlink():
            raise SystemExit(f"{p} is a symbolic link; the base directory must be plain (MM-20)")
    on_disk = sorted(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file())
    if on_disk != sorted(files):
        raise SystemExit(f"{base} does not hold exactly the files of {MODELS_JSON} [{BASE_NAME}]")
    if verify:
        for rel, want in files.items():
            if sha256_file(base / rel) != want["sha256"]:
                raise SystemExit(f"{base / rel} differs from its SHA-256 in {MODELS_JSON}")
    rev = record["revision"]
    if len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev):
        raise SystemExit(f"base revision {rev!r} is not a 40-hex commit (MM-18)")
    return {"repo": record["repo"], "revision": rev, "local_dir": str(base),
            "files": {k: v["sha256"] for k, v in files.items()},
            "files_sha256": sha256_bytes(canonical({k: v["sha256"] for k, v in files.items()})),
            "verified": verify}


def read_prompts(prompts_dir: Path) -> tuple[list[dict], Path]:
    path = prompts_dir / "train.jsonl"
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ids = [r["prompt_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: prompt ids are not unique")
    return rows, path


# ---------------------------------------------------------------------------
# Cuts
# ---------------------------------------------------------------------------

def canonicalize_safetensors_header(path: Path) -> None:
    """safetensors serializes its header (tensor table and `__metadata__`) from hash maps, so two saves of identical
    tensors can differ in key order (measured: same tensor bytes, different file SHA-256). Rewrite the header as
    JSON with sorted keys, compact separators and space padding to 8 bytes (the format allows trailing spaces); the
    tensor bytes and offsets are unchanged, so the file hash depends on the weights and metadata only."""
    import struct

    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    meta = header.get("__metadata__", {})
    if "lora_adapter_metadata" in meta:
        # LoraConfig.target_modules is a set; diffusers writes list(set) (loaders/peft.py save_lora_adapter), whose
        # order varies between processes. Sorted here; the loader reads it back as a set.
        cfg = json.loads(meta["lora_adapter_metadata"])
        if isinstance(cfg.get("target_modules"), list):
            cfg["target_modules"] = sorted(cfg["target_modules"])
        meta["lora_adapter_metadata"] = json.dumps(cfg, indent=2, sort_keys=True)
    text = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    text += b" " * (-len(text) % 8)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text + raw[8 + n:])
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_cut(unet, out: Path, epoch: int, global_step: int, samples_consumed: int, base_ref: dict, sampler: dict,
             lora_cfg: dict, tracker, extra: dict) -> Path:
    """checkpoint-<epoch>/ via .checkpoint-<epoch>.partial/ and an atomic rename: the LoRA weights
    (`save_lora_adapter`, safetensors with the LoraConfig in the header), adapter_config.json, sampler.json (the
    deployment sampler), base_ref.json, trainer_state.json (resume-only information), then cut.json last with the
    SHA-256 of every other file."""
    cut = out / f"checkpoint-{epoch}"
    if cut.exists():
        return cut
    tmp = out / f".checkpoint-{epoch}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    unet.save_lora_adapter(str(tmp), adapter_name="default", safe_serialization=True, weight_name=LORA_WEIGHTS)
    canonicalize_safetensors_header(tmp / LORA_WEIGHTS)
    os.chmod(tmp / LORA_WEIGHTS, 0o644)              # safetensors writes 0600; the other files are 0644
    write_json(tmp / "adapter_config.json", {**lora_cfg, "base_model_name_or_path": base_ref["repo"],
                                             "base_model_revision": base_ref["revision"], "format": "peft",
                                             "loader": "UNet2DConditionModel.load_lora_adapter(dir, prefix=None)"})
    write_json(tmp / "sampler.json", sampler)
    write_json(tmp / "base_ref.json", {k: base_ref[k] for k in ("repo", "revision", "files", "files_sha256")})
    write_json(tmp / "trainer_state.json", {
        "epoch": epoch, "global_step": global_step, "samples_consumed": samples_consumed,
        "stat_tracker": {k: [float(x) for x in v] for k, v in sorted(tracker.stats.items())},
        "note": "resume-only information; the run does not resume from a cut (a preempted run restarts)"})
    files = {name: sha256_file(tmp / name) for name in CUT_FILES}
    write_json(tmp / CUT_MARKER, {
        "schema": "ptc_mm.ddpo_sd15_cut", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "epoch": epoch, "global_step": global_step, "samples_consumed": samples_consumed,
        "files": files, "deployable": list(DEPLOYABLE), "created_at": utc_now(), **extra,
    })
    os.replace(tmp, cut)
    return cut


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--profile", choices=sorted(PROFILES), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--seed", type=int, default=3000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, help="override the profile's epoch count")
    p.add_argument("--no-verify-base", action="store_true", help="skip re-hashing the base files (host checks)")
    p.add_argument("--train-batch", type=int, help="memory knob: override the profile's train batch")
    p.add_argument("--no-gradient-checkpointing", action="store_true", help="memory knob: faster, more memory")
    return p.parse_args(argv)


def resolve_settings(args) -> dict:
    s = {**COMMON, **PROFILES[args.profile]}
    if args.epochs is not None:
        s["epochs"] = args.epochs
    if args.train_batch is not None:
        s["train_batch"] = args.train_batch
    if args.no_gradient_checkpointing:
        s["gradient_checkpointing"] = False
    s["seed"] = args.seed
    s["samples_per_epoch"] = s["prompts_per_epoch"] * s["images_per_prompt"]
    if s["samples_per_epoch"] % s["sample_batch"] or s["samples_per_epoch"] % s["train_batch"]:
        raise SystemExit("samples per epoch must be a multiple of the sample and train batch sizes")
    # ddpo-pytorch accumulates over train batches and timesteps (scripts/train.py:70-79); here the samples of one
    # epoch are split into `updates_per_epoch` optimizer steps (config/dgx.py: 256 samples, 4 updates per epoch).
    if s["samples_per_epoch"] % (s["train_batch"] * s["updates_per_epoch"]):
        raise SystemExit("samples per epoch must be a multiple of train_batch x updates_per_epoch")
    s["grad_accum_batches"] = s["samples_per_epoch"] // (s["train_batch"] * s["updates_per_epoch"])
    s["num_train_timesteps"] = max(1, int(s["num_steps"] * s["timestep_fraction"]))
    return s


def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.out.resolve()
    attempt = prepare_out(out)
    runtime = write_env_freeze(out)                 # before torch is imported (MM-33)
    s = resolve_settings(args)
    t_start = time.time()
    base_ref = base_binding(args.base.resolve(), verify=not args.no_verify_base)
    prompts, prompts_file = read_prompts(args.prompts)

    import numpy as np
    import torch
    from peft import LoraConfig

    device = torch.device(args.device)
    dtype = torch.float16 if s["precision"] == "fp16" else torch.float32
    if s["allow_tf32"] and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(derive_seed("torch", s["seed"]))   # nothing below draws from it; recorded for completeness

    sd = load_sd15(args.base, device, dtype)
    unet = sd["unet"]
    lora_config = LoraConfig(**LORA)
    unet.add_adapter(lora_config)
    for prm in unet.parameters():                    # LoRA parameters in fp32 (diffusers cast_training_params)
        if prm.requires_grad:
            prm.data = prm.data.float()
    if s["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    params = [prm for prm in unet.parameters() if prm.requires_grad]
    lora_cfg = json.loads(json.dumps(lora_config.to_dict(), default=lambda o: sorted(o) if isinstance(o, set)
                                     else str(o)))
    optimizer = torch.optim.AdamW(params, lr=s["learning_rate"], betas=(s["adam_beta1"], s["adam_beta2"]),
                                  weight_decay=s["adam_weight_decay"], eps=s["adam_epsilon"])
    scaler = torch.amp.GradScaler(device.type, enabled=(s["precision"] == "fp16"))
    neg = encode_prompts(sd, [""], device)
    sampler = sampler_doc(s["num_steps"], s["eta"], s["guidance_scale"], s["resolution"],
                          dict(sd["scheduler"].config))
    tracker = PerPromptStatTracker(s["stat_buffer_size"], s["stat_min_count"])

    run_args = {
        "schema": "ptc_mm.ddpo_sd15_run_args", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "profile": args.profile, "attempt": attempt, "settings": s, "lora": lora_cfg,
        "seed": s["seed"],
        "seed_derivation": ("derive_seed(*parts) = first 8 bytes of SHA-256(':'.join(parts)) >> 1. Epoch e prompts: "
                            "numpy Generator(PCG64(derive_seed('prompts', seed, e))).choice(n_train, P, "
                            "replace=False); sample i of epoch e: noise seed derive_seed('sample', seed, e, i) for a "
                            "CPU torch.Generator (initial latents, then one draw per denoising step); training "
                            "order: torch.Generator('cpu').manual_seed(derive_seed('perm', seed, e, inner)) for the "
                            "sample permutation and the per-sample timestep permutations; LoRA init: torch global RNG "
                            "seeded with derive_seed('torch', seed) (peft gaussian init draws from it)"),
        "base_model": base_ref["local_dir"], "base_ref": {k: base_ref[k] for k in ("repo", "revision",
                                                                                   "files_sha256", "verified")},
        "train_file": str(prompts_file), "train_file_sha256": sha256_file(prompts_file),
        "n_train_prompts": len(prompts),
        "sampler": sampler, "sampler_sha256": sha256_bytes(canonical(sampler)),
        "consumption_log": "samples.jsonl", "step_log": "train_steps.jsonl",
        "consumption_semantics": ("samples.jsonl: one line per generated sample at sampling time; every sample of "
                                  "epoch e is consumed by the optimizer steps of epoch e only; train_steps.jsonl "
                                  "names, per optimizer step, the sample ids whose gradients it applied"),
        "device": str(device), "dtype": str(dtype),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "sampler_code_sha256": runtime["sampler_code_sha256"], "script_sha256": runtime["script_sha256"],
        "created_at": utc_now(),
    }
    write_json(out / "run_args.json", run_args)

    global_step = 0
    samples_consumed = 0
    P, G, B = s["prompts_per_epoch"], s["images_per_prompt"], s["sample_batch"]
    for epoch in range(1, s["epochs"] + 1):
        t_ep = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        # ---------------- sampling ----------------
        unet.eval()
        rng = np.random.Generator(np.random.PCG64(derive_seed("prompts", s["seed"], epoch)))
        chosen = rng.choice(len(prompts), size=P, replace=False)
        plan = [(prompts[int(c)], g) for c in chosen for g in range(G)]
        first_step = global_step + 1
        last_step = global_step + s["updates_per_epoch"] * s["num_inner_epochs"]
        rec = {"latents": [], "next_latents": [], "log_probs": [], "embeds": [], "timesteps": None}
        rewards, sample_ids, prompt_ids = [], [], []
        for b0 in range(0, len(plan), B):
            batch = plan[b0:b0 + B]
            idx = list(range(b0, b0 + len(batch)))
            seeds = [derive_seed("sample", s["seed"], epoch, i) for i in idx]
            embeds = encode_prompts(sd, [row["prompt"] for row, _ in batch], device)
            negb = neg.expand(len(batch), -1, -1)
            imgs, lat, logp, ts = sample_with_logprob(sd, unet, embeds, negb, seeds, s["num_steps"], s["eta"],
                                                      s["guidance_scale"], s["resolution"], device)
            rec["latents"].append(lat[:, :-1])
            rec["next_latents"].append(lat[:, 1:])
            rec["log_probs"].append(logp)
            rec["embeds"].append(embeds)
            rec["timesteps"] = ts
            for k, (row, g) in enumerate(batch):
                kb = jpeg_kb(to_uint8(imgs[k]), s["jpeg_quality"])
                sid = sha256_bytes(canonical(["ddpo-sd15", s["seed"], attempt, epoch, idx[k], row["prompt_id"],
                                              seeds[k], g]))
                rewards.append(-kb)
                sample_ids.append(sid)
                prompt_ids.append(row["prompt_id"])
                append_jsonl(out / "samples.jsonl", {
                    "epoch": epoch, "sample_index": idx[k], "sample_id": sid, "prompt_id": row["prompt_id"],
                    "prompt": row["prompt"], "member": g, "seed": seeds[k], "jpeg_kb": kb, "reward": -kb,
                    "steps": [first_step, last_step], "attempt": attempt, "utc": utc_now()})
        t_sample = time.time() - t_ep
        adv = tracker.update(prompt_ids, rewards)
        n = len(plan)
        lat_all = torch.cat(rec["latents"])
        next_all = torch.cat(rec["next_latents"])
        logp_all = torch.cat(rec["log_probs"])
        emb_all = torch.cat(rec["embeds"])
        adv_all = torch.as_tensor(adv, dtype=torch.float32, device=device)
        T = s["num_steps"]
        ts_all = rec["timesteps"].to(device)
        # ---------------- training (scripts/train.py:455-593) ----------------
        unet.train()
        t_tr = time.time()
        for inner in range(s["num_inner_epochs"]):
            g = torch.Generator("cpu").manual_seed(derive_seed("perm", s["seed"], epoch, inner))
            perm = torch.randperm(n, generator=g)
            tperm = torch.stack([torch.randperm(T, generator=g) for _ in range(n)])
            order = perm.tolist()
            batches = [order[i:i + s["train_batch"]] for i in range(0, n, s["train_batch"])]
            info = {"loss": [], "approx_kl": [], "clipfrac": []}
            consumed = []
            optimizer.zero_grad(set_to_none=True)
            for bi, bidx in enumerate(batches):
                bt = torch.tensor(bidx)
                negb = neg.expand(len(bidx), -1, -1)
                for j in range(s["num_train_timesteps"]):
                    tj = tperm[bt, j]
                    lat_j = lat_all[bt.to(device), tj.to(device)]
                    nxt_j = next_all[bt.to(device), tj.to(device)]
                    old_lp = logp_all[bt.to(device), tj.to(device)]
                    t_j = ts_all[tj.to(device)]
                    eps = predict_noise(unet, lat_j, t_j, emb_all[bt.to(device)], negb, s["guidance_scale"])
                    _, lp = ddim_step_with_logprob(sd["scheduler"], eps, t_j, lat_j, s["eta"], prev_sample=nxt_j)
                    a = torch.clamp(adv_all[bt.to(device)], -s["adv_clip_max"], s["adv_clip_max"])
                    ratio = torch.exp(lp - old_lp)
                    unclipped = -a * ratio
                    clipped = -a * torch.clamp(ratio, 1.0 - s["clip_range"], 1.0 + s["clip_range"])
                    loss = torch.mean(torch.maximum(unclipped, clipped))
                    scaler.scale(loss / (s["grad_accum_batches"] * s["num_train_timesteps"])).backward()
                    info["loss"].append(float(loss.detach()))
                    info["approx_kl"].append(float(0.5 * torch.mean((lp - old_lp) ** 2)))
                    info["clipfrac"].append(float(torch.mean((torch.abs(ratio - 1.0) > s["clip_range"]).float())))
                consumed.extend(sample_ids[i] for i in bidx)
                if (bi + 1) % s["grad_accum_batches"] == 0:
                    global_step += 1
                    append_jsonl(out / "train_steps.jsonl", {"step": global_step, "epoch": epoch, "inner": inner,
                                                             "sample_ids": consumed, "utc": utc_now()})
                    scaler.unscale_(optimizer)
                    gnorm = float(torch.nn.utils.clip_grad_norm_(params, s["max_grad_norm"]))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    append_jsonl(out / "train_log.jsonl", {
                        "step": global_step, "epoch": epoch, "grad_norm": gnorm, "scale": float(scaler.get_scale()),
                        **{k: float(np.mean(v)) for k, v in info.items()}})
                    info = {"loss": [], "approx_kl": [], "clipfrac": []}
                    consumed = []
        samples_consumed += n
        t_train = time.time() - t_tr
        r = np.array(rewards)
        mem = {"epoch": epoch, "seconds_sampling": round(t_sample, 2), "seconds_training": round(t_train, 2),
               "seconds_epoch": round(time.time() - t_ep, 2), "global_step": global_step,
               "reward_mean": float(r.mean()), "reward_std": float(r.std()), "jpeg_kb_mean": float(-r.mean()),
               "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1), "utc": utc_now()}
        if device.type == "cuda":
            mem["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
            mem["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
        append_jsonl(out / "memory.jsonl", mem)
        print(json.dumps(mem, sort_keys=True), flush=True)
        if epoch % s["save_every"] == 0 or epoch == s["epochs"]:
            save_cut(unet, out, epoch, global_step, samples_consumed, base_ref, sampler, lora_cfg, tracker,
                     {"script_sha256": runtime["script_sha256"], "sampler_code_sha256": runtime["sampler_code_sha256"],
                      "seed": s["seed"], "profile": args.profile})
    append_jsonl(out / "attempts.jsonl", {"attempt": attempt, "finished_at": utc_now(), "epochs": s["epochs"],
                                          "global_step": global_step, "samples_consumed": samples_consumed,
                                          "wall_seconds": round(time.time() - t_start, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
