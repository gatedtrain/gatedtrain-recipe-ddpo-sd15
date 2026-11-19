#!/usr/bin/env python
"""Per-item image evaluation of an SD1.5 + LoRA cut (or the base alone) for prototype area "image generation, SD1.5"
(MULTIMODAL-SPEC section 6 row 2a; prototype, unpromotable).

An item is one registered prompt (design 05 section 3.3): the first name of an ImageNet animal class held out from
training (README.md "Items"). Per item and repeat r: one image from the deployment sampler (the cut's sampler.json:
DDIM, eta, steps, CFG, resolution) with noise seed derive_seed('eval', eval_seeds[r], item), the same for every
arm (common random numbers). Every image is kept as PNG with its SHA-256; every metric is computed from the saved
uint8 image.

Metrics (README.md "Metrics and tiers"; the tiers were proposals, settled in "Decisions taken"):
  webp_lossless_bpp   deciding (proposed T3), lower is better, [0, null]: bits per pixel of the image under
                      libwebp lossless (Pillow `WEBP`, lossless=True, quality=100, method=4, exact=True)
  jpeg_q95_kb         supporting, T0 (the training reward, ddpo-pytorch rewards.py:7-17), lower is better, [0, null]
  prompt_violation    constraint c1 (non-inferiority constraint clause, 05 section 3.9), lower is better, {0, 1}:
                      1 when SigLIP so400m zero-shot over the 398 animal labels does not rank the prompted animal
                      among its top 5
Descriptive sub-metrics only (never deciding): png_bpp, aesthetic (LAION improved-aesthetic-predictor on CLIP
ViT-L/14), siglip_rank, siglip_top1, nonfinite.

Ports from kvablack/ddpo-pytorch at commit 1958463f020112c9a7bc85768d296daacc2e1b4b (MIT License,
Copyright (c) 2023 Kevin Black), cited by file:line of that commit where they are used: the shared sampler block
(the DDIM step with log-probability and the sampling loop with classifier-free guidance, byte-identical in
train_ddpo.py), the JPEG-compressibility measurement (rewards.py), the animal-class split (prompts.py) and the
aesthetic scorer (aesthetic_scorer.py).

Two modes:
  python eval_images.py --make-items --classes data-source/imagenet_classes.txt --n-items 150 --split-seed S
      writes prompts/train.jsonl (training shard), items/items.jsonl (registered items, disjoint),
      constraints/constraint_set.json, and their meta files
  python eval_images.py (--checkpoint CUT | --base-only) --items DIR --out DIR --eval-seeds 11,12
                        [--model-hash H] [--base-model-hash H] [--item-set-hash H] [--limit N] [--device cuda]
      writes DIR/images/<item>/r<repeat>.png, DIR/scores.jsonl (every metric), and one MM-1/MM-2 arm per metric:
      DIR/arm-<metric>/{items.jsonl, eval_meta.json, images/ (hard links)}

The arm needs gatedtrain hashes this runtime does not compute (the adapter's `model_hash`, MM-17/MM-18, and the item
set's manifest hash, MM-6): pass them, or let gatedtrain fill `eval_meta.json` (README.md "What gatedtrain needs").
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "models" / "stable-diffusion-v1-5"
SIGLIP_DIR = HERE / "models" / "siglip-so400m-patch14-384"
CLIP_DIR = HERE / "models" / "clip-vit-large-patch14"
AESTHETIC_MLP = HERE / "models" / "aesthetic-mlp" / "sac+logos+ava1-l14-linearMSE.safetensors"
MODELS_JSON = HERE / "models" / "MODELS.json"
CONSTRAINTS = HERE / "constraints" / "constraint_set.json"
LORA_WEIGHTS = "pytorch_lora_weights.safetensors"
ADAPTER = "ddpo_sd15"
SCORER_VERSION = "eval_images@0.1"
N_ANIMAL_CLASSES = 398          # ImageNet classes 0-397 (ddpo-pytorch prompts.py:35-36 `imagenet_animals`)
ITEM_NAMESPACE = 0x5D15A3       # split permutation: SeedSequence([ITEM_NAMESPACE, split_seed])
SAMPLER_DEFAULT = {"num_steps": 20, "eta": 1.0, "guidance_scale": 5.0, "resolution": 512}
WEBP = {"format": "WEBP", "lossless": True, "quality": 100, "method": 4, "exact": True}
PNG = {"format": "PNG", "compress_level": 9, "optimize": False}
TOP_K = 5
LABEL_TEMPLATE = "a photo of a {}."
METRICS = {
    # metric -> evaluator id, direction, range, tier (proposal), role
    "webp_lossless_bpp": ("t2i.sd15_webp_items@0.1", "lower_is_better", [0, None], "T3 (proposed)", "deciding"),
    "jpeg_q95_kb": ("t2i.sd15_jpeg_items@0.1", "lower_is_better", [0, None], "T0", "supporting"),
    "prompt_violation": ("t2i.sd15_animal_presence@0.1", "lower_is_better", [0, 1], "T2 (transfer test pending)",
                         "constraint"),
}
ERROR_CLASSES = ("scorer_error",)


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
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


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


# ---------------------------------------------------------------------------
# Items, training prompts, constraint set
# ---------------------------------------------------------------------------

def class_names(classes_file: Path) -> list[str]:
    lines = classes_file.read_text(encoding="utf-8").splitlines()[:N_ANIMAL_CLASSES]
    names = [ln.split(",")[0].strip() for ln in lines]
    if len(set(names)) != len(names):
        raise SystemExit("first names of the animal classes are not unique")
    return names


def make_items(classes_file: Path, n_items: int, split_seed: int) -> int:
    import numpy as np

    names = class_names(classes_file)
    targets = [HERE / "prompts", HERE / "items", HERE / "constraints"]
    for t in targets:
        if t.exists() and any(t.iterdir()):
            raise SystemExit(f"{t} exists and is not empty")
        t.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(np.random.SeedSequence([ITEM_NAMESPACE, split_seed]))
    perm = [int(x) for x in rng.permutation(N_ANIMAL_CLASSES)]
    held = sorted(perm[:n_items])
    train = sorted(perm[n_items:])
    items = [{"item": f"inet-{c:04d}", "prompt": names[c], "class_index": c, "label": names[c]} for c in held]
    shard = [{"prompt_id": f"inet-{c:04d}", "prompt": names[c], "class_index": c} for c in train]
    overlap = {r["prompt"] for r in items} & {r["prompt"] for r in shard}
    if overlap:
        raise SystemExit(f"exact overlap between items and training prompts: {sorted(overlap)[:5]}")
    write_jsonl(HERE / "items" / "items.jsonl", items)
    write_jsonl(HERE / "prompts" / "train.jsonl", shard)
    source = {"file": "data-source/imagenet_classes.txt", "sha256": sha256_file(classes_file),
              "origin": "kvablack/ddpo-pytorch@1958463f020112c9a7bc85768d296daacc2e1b4b "
                        "ddpo_pytorch/assets/imagenet_classes.txt (lines 1-398 = classes 0-397, prompts.py:35-36)",
              "prompt_rule": "the first comma-separated name of the class line"}
    split = {"n_classes": N_ANIMAL_CLASSES, "n_items": len(items), "n_train": len(shard), "split_seed": split_seed,
             "generator": f"numpy default_rng(SeedSequence([{ITEM_NAMESPACE:#x}, split_seed])).permutation(398); the "
                          "first n_items classes are held out as items, the rest form the training shard",
             "numpy": np.__version__, "exact_overlap": 0}
    write_json(HERE / "items" / "item_set_meta.json", {
        "schema": "ptc_mm.sd15_item_set", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "source": source, "split": split, "items_sha256": sha256_file(HERE / "items" / "items.jsonl"),
        "train_sha256": sha256_file(HERE / "prompts" / "train.jsonl")})
    write_json(HERE / "prompts" / "prompts_meta.json", {
        "schema": "ptc_mm.sd15_train_prompts", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "source": source, "split": split, "train_sha256": sha256_file(HERE / "prompts" / "train.jsonl")})
    models = json.loads(MODELS_JSON.read_text(encoding="utf-8"))["models"]
    sig = models["siglip-so400m-patch14-384"]
    constraint_set = {
        "schema": "ptc_mm.constraint_set", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "constraint_set_id": "sd15-animal-presence-v1",
        "constraints": [{
            "id": "c1",
            "text": "The output must not fail to depict the animal named in the prompt.",
            "checker": {
                "kind": "tool", "name": "siglip_zero_shot_topk", "k": TOP_K,
                "model": {"repo": sig["repo"], "revision": sig["revision"],
                          "weights_sha256": sig["files"]["model.safetensors"]["sha256"], "backbone_family": "siglip"},
                "labels": names, "label_template": LABEL_TEMPLATE,
                "text_preprocess": "SiglipProcessor, padding='max_length', max_length=64",
                "image_preprocess": "SiglipProcessor image processor from the model directory (384 x 384)",
                "rule": "violation iff the prompted class's label is not among the k labels with the highest "
                        "image-text similarity (cosine of the normalized SigLIP embeddings)",
            }}],
        "judges": [],
        "canaries": [],
    }
    write_json(CONSTRAINTS, constraint_set)
    print(json.dumps({"items": len(items), "train": len(shard), "exact_overlap": 0,
                      "constraint_set_sha256": sha256_bytes(canonical(constraint_set))}, indent=2))
    return 0


def read_items(items_dir: Path) -> list[dict]:
    rows = [json.loads(ln) for ln in (items_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    ids = [r["item"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("item ids are not unique")
    return rows


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def encoded_bpp(arr, params: dict) -> float:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, **params)
    return buf.tell() * 8 / (arr.shape[0] * arr.shape[1])


class SiglipChecker:
    def __init__(self, labels: list[str], device, dtype):
        import torch
        from transformers import AutoProcessor, SiglipModel

        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(str(SIGLIP_DIR), local_files_only=True)
        self.model = SiglipModel.from_pretrained(str(SIGLIP_DIR), torch_dtype=dtype, local_files_only=True).to(device)
        self.model.eval()
        self.labels = labels
        with torch.no_grad():
            text = self.processor(text=[LABEL_TEMPLATE.format(n) for n in labels], padding="max_length",
                                  max_length=64, return_tensors="pt")
            feats = self.model.get_text_features(input_ids=text["input_ids"].to(device)).pooler_output
            self.text = torch.nn.functional.normalize(feats.float(), dim=-1)

    def __call__(self, pil_images, label_indices):
        torch = self.torch
        with torch.no_grad():
            px = self.processor(images=pil_images, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            img = torch.nn.functional.normalize(self.model.get_image_features(pixel_values=px).pooler_output.float(),
                                                dim=-1)
            sims = img @ self.text.T
        out = []
        for row, target in zip(sims, label_indices):
            order = torch.argsort(row, descending=True).tolist()
            rank = order.index(target) + 1
            out.append({"siglip_rank": rank, "siglip_top1": self.labels[order[0]],
                        "violation": int(rank > TOP_K)})
        return out


class AestheticScorer:
    """ddpo-pytorch aesthetic_scorer.py:13-53 (LAION improved-aesthetic-predictor on CLIP ViT-L/14 image
    embeddings, L2-normalized), with the MLP head read from safetensors."""

    def __init__(self, device, dtype):
        import torch
        from safetensors.torch import load_file
        from transformers import CLIPImageProcessor, CLIPModel

        self.torch = torch
        self.device, self.dtype = device, dtype
        self.processor = CLIPImageProcessor.from_pretrained(str(CLIP_DIR), local_files_only=True)
        self.clip = CLIPModel.from_pretrained(str(CLIP_DIR), torch_dtype=dtype, local_files_only=True).to(device)
        self.clip.eval()
        mlp = torch.nn.Sequential(torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2), torch.nn.Linear(1024, 128),
                                  torch.nn.Dropout(0.2), torch.nn.Linear(128, 64), torch.nn.Dropout(0.1),
                                  torch.nn.Linear(64, 16), torch.nn.Linear(16, 1))
        state = {k.removeprefix("layers."): v for k, v in load_file(str(AESTHETIC_MLP)).items()}
        mlp.load_state_dict(state)
        self.mlp = mlp.to(device).eval()

    def __call__(self, pil_images):
        torch = self.torch
        with torch.no_grad():
            px = self.processor(images=pil_images, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            emb = self.clip.get_image_features(pixel_values=px).pooler_output.float()
            emb = emb / torch.linalg.vector_norm(emb, dim=-1, keepdim=True)
            return [float(x) for x in self.mlp(emb).squeeze(1)]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def check_cut(cut: Path) -> dict:
    marker = json.loads((cut / "cut.json").read_text(encoding="utf-8"))
    for name, want in marker["files"].items():
        if sha256_file(cut / name) != want:
            raise SystemExit(f"{cut / name} differs from its SHA-256 in cut.json")
    if marker.get("sampler_code_sha256") != sampler_code_sha256():
        raise SystemExit("the cut was trained with another sampler code than this evaluator's")
    base_ref = json.loads((cut / "base_ref.json").read_text(encoding="utf-8"))
    record = json.loads(MODELS_JSON.read_text(encoding="utf-8"))["models"]["stable-diffusion-v1-5"]
    if base_ref["revision"] != record["revision"] or base_ref["files"] != {k: v["sha256"] for k, v in
                                                                           record["files"].items()}:
        raise SystemExit("the cut names another base model than models/stable-diffusion-v1-5")
    return marker


def resolve_sampler(args, cut: Path | None, scheduler_config: dict) -> dict:
    want = {k: getattr(args, k) for k in SAMPLER_DEFAULT if getattr(args, k) is not None}
    if cut is not None:
        doc = json.loads((cut / "sampler.json").read_text(encoding="utf-8"))
        for k, v in want.items():
            if doc[k] != v:
                raise SystemExit(f"--{k.replace('_', '-')} {v} differs from the cut's sampler.json ({doc[k]})")
        if doc != sampler_doc(doc["num_steps"], doc["eta"], doc["guidance_scale"], doc["resolution"],
                              scheduler_config):
            raise SystemExit("the cut's sampler.json is not this evaluator's sampler")
        return doc
    s = {**SAMPLER_DEFAULT, **want}
    return sampler_doc(s["num_steps"], s["eta"], s["guidance_scale"], s["resolution"], scheduler_config)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        dst.write_bytes(src.read_bytes())


def evaluate(args) -> int:
    import numpy as np
    import torch
    from PIL import Image, features

    t0 = time.time()
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    cut = args.checkpoint.resolve() if args.checkpoint else None
    marker = check_cut(cut) if cut else None
    items = read_items(args.items)
    if args.limit:
        items = items[: args.limit]
    seeds = [int(x) for x in args.eval_seeds.split(",")]
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    constraint_set = json.loads(CONSTRAINTS.read_text(encoding="utf-8"))
    labels = constraint_set["constraints"][0]["checker"]["labels"]
    label_index = {n: i for i, n in enumerate(labels)}

    sd = load_sd15(args.base, device, dtype)
    if cut is not None:
        sd["unet"].load_lora_adapter(str(cut), prefix=None, weight_name=LORA_WEIGHTS, adapter_name="ddpo",
                                     local_files_only=True)
    sd["unet"].eval()
    sampler = resolve_sampler(args, cut, dict(sd["scheduler"].config))
    siglip = SiglipChecker(labels, device, dtype)
    aesthetic = AestheticScorer(device, dtype) if not args.no_aesthetic else None
    neg = encode_prompts(sd, [""], device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    jobs = [(r, it) for r in range(len(seeds)) for it in items]
    scores = []
    for b0 in range(0, len(jobs), args.batch_size):
        batch = jobs[b0:b0 + args.batch_size]
        noise_seeds = [derive_seed("eval", seeds[r], it["item"]) for r, it in batch]
        embeds = encode_prompts(sd, [it["prompt"] for _, it in batch], device)
        imgs, _, _, _ = sample_with_logprob(sd, sd["unet"], embeds, neg.expand(len(batch), -1, -1), noise_seeds,
                                            sampler["num_steps"], sampler["eta"], sampler["guidance_scale"],
                                            sampler["resolution"], device)
        pils, rows = [], []
        for k, (r, it) in enumerate(batch):
            img = imgs[k]
            nonfinite = not bool(torch.isfinite(img).all())
            if nonfinite:
                img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            arr = to_uint8(img)
            rel = f"images/{it['item']}/r{r}.png"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(out / rel, format="PNG")
            pil = Image.open(out / rel).convert("RGB")
            pils.append(pil)
            row = {"item": it["item"], "repeat": r, "seed": seeds[r], "noise_seed": noise_seeds[k],
                   "output_ref": {"path": rel, "sha256": sha256_file(out / rel)}, "nonfinite": nonfinite,
                   "errors": {}}
            saved = np.asarray(pil)
            for name, fn in (("webp_lossless_bpp", lambda a: encoded_bpp(a, WEBP)),
                             ("png_bpp", lambda a: encoded_bpp(a, PNG)), ("jpeg_q95_kb", jpeg_kb)):
                try:
                    row[name] = fn(saved)
                except Exception as e:                          # scorer failure: an error record (MM-4)
                    row[name] = None
                    row["errors"][name] = {"class": "scorer_error", "message": f"{type(e).__name__}: {e}"[:300]}
            rows.append(row)
        try:
            sig = siglip(pils, [label_index[it["label"]] for _, it in batch])
        except Exception as e:
            sig = [None] * len(batch)
            for row in rows:
                row["errors"]["prompt_violation"] = {"class": "scorer_error",
                                                     "message": f"{type(e).__name__}: {e}"[:300]}
        aes = None
        if aesthetic is not None:
            try:
                aes = aesthetic(pils)
            except Exception:
                aes = None
        for k, row in enumerate(rows):
            row.update(sig[k] or {"siglip_rank": None, "siglip_top1": None, "violation": None})
            row["prompt_violation"] = row.pop("violation")
            row["aesthetic"] = aes[k] if aes else None
            scores.append(row)
        print(f"{len(scores)}/{len(jobs)} images, {time.time() - t0:.0f} s", flush=True)
    wall = time.time() - t0
    write_jsonl(out / "scores.jsonl", scores)

    code_sha = sha256_file(Path(__file__).resolve())
    sampler_hash = sha256_bytes(canonical(sampler))
    models = json.loads(MODELS_JSON.read_text(encoding="utf-8"))["models"]
    pil_info = {"pillow": importlib.metadata.version("pillow"), "libwebp": features.version("webp"),
                "zlib": features.version("zlib"), "libjpeg": features.version("jpg")}
    scorers = {
        "webp_lossless_bpp": {"name": "webp_lossless_bpp", "version": f"{SCORER_VERSION} pillow "
                              f"{pil_info['pillow']} libwebp {pil_info['libwebp']} {json.dumps(WEBP, sort_keys=True)}",
                              "weights_sha256": None, "code_sha256": code_sha},
        "jpeg_q95_kb": {"name": "jpeg_q95_kb", "version": f"{SCORER_VERSION} pillow {pil_info['pillow']} libjpeg "
                        f"{pil_info['libjpeg']} quality 95", "weights_sha256": None, "code_sha256": code_sha},
        "prompt_violation": {"name": "siglip_zero_shot_top5", "version": f"{SCORER_VERSION} "
                             f"{models['siglip-so400m-patch14-384']['repo']}@"
                             f"{models['siglip-so400m-patch14-384']['revision']}",
                             "weights_sha256": models["siglip-so400m-patch14-384"]["files"]["model.safetensors"]
                             ["sha256"], "code_sha256": code_sha},
    }
    for metric, (evaluator_id, direction, rng, tier, role) in METRICS.items():
        arm = out / f"arm-{metric}"
        arm.mkdir()
        lines = []
        for row in scores:
            link_or_copy(out / row["output_ref"]["path"], arm / row["output_ref"]["path"])
            err = row["errors"].get(metric)
            score = None if err else row[metric]
            if metric == "prompt_violation":
                sub = None if err else {"violated": ["c1"] if score else [], "per_constraint": {"c1": score},
                                        "siglip_rank": row["siglip_rank"], "siglip_top1": row["siglip_top1"]}
            else:
                sub = {k: row[k] for k in ("webp_lossless_bpp", "png_bpp", "jpeg_q95_kb", "aesthetic",
                                           "siglip_rank", "prompt_violation") if row.get(k) is not None}
                sub["nonfinite"] = int(row["nonfinite"])
            lines.append({"item": row["item"], "repeat": row["repeat"], "seed": row["seed"], "score": score,
                          "error": err, "sub_metrics": sub, "output_ref": row["output_ref"]})
        write_jsonl(arm / "items.jsonl", lines)
        meta = {
            "schema": "gatedtrain.item_eval", "schema_version": "1.0",
            "evaluator_id": evaluator_id, "adapter": ADAPTER, "metric": metric, "direction": direction,
            "score_range": rng,
            "checkpoint": {"model_hash": args.model_hash, "base_model_hash": args.base_model_hash},
            "normalizer_sha256": None, "env_version_sha256": None,
            "item_set_hash": args.item_set_hash, "eval_seeds": seeds,
            "sampler": sampler, "sampler_hash": sampler_hash,
            "scorer": scorers[metric],
            "framework": {"name": "diffusers", "version": importlib.metadata.version("diffusers")},
            "created_at": utc_now(),
        }
        if metric == "prompt_violation":
            meta["constraint_set_hash"] = sha256_bytes(canonical(constraint_set))
            meta["model_manifest_hash"] = args.model_hash
        write_json(arm / "eval_meta.json", meta)

    def stats(vals):
        vals = [v for v in vals if v is not None]
        n = len(vals)
        if not n:
            return {"n": 0}
        mean = sum(vals) / n
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else None
        return {"n": n, "mean": mean, "sd": sd, "min": min(vals), "max": max(vals),
                "distinct": len({round(v, 9) for v in vals})}

    summary = {"checkpoint": str(cut) if cut else None, "base_only": cut is None,
               "epoch": marker["epoch"] if marker else 0, "items": len(items), "repeats": len(seeds),
               "images": len(scores), "errors": sum(1 for r in scores if r["errors"]),
               "nonfinite": sum(int(r["nonfinite"]) for r in scores),
               "metrics": {m: stats([r[m] for r in scores]) for m in ("webp_lossless_bpp", "png_bpp",
                                                                       "jpeg_q95_kb", "prompt_violation",
                                                                       "aesthetic")},
               "wall_seconds": round(wall, 1), "seconds_per_image": round(wall / max(1, len(scores)), 3),
               "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
               "sampler_hash": sampler_hash, "device": str(device), "dtype": str(dtype),
               "missing_hashes": [k for k, v in (("model_hash", args.model_hash),
                                                  ("item_set_hash", args.item_set_hash)) if v is None]}
    if device.type == "cuda":
        summary["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
        summary["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
    lock = HERE / "uv.lock"
    write_json(out / "env-runtime.json", {
        "python": sys.version.split()[0], "executable": sys.executable,
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else None, "script_sha256": code_sha,
        "sampler_code_sha256": sampler_code_sha256(), "pil": pil_info, "torch": torch.__version__,
        "checkpoint": str(cut) if cut else None, "items": str(args.items.resolve()),
        "items_sha256": sha256_file(args.items / "items.jsonl"), "constraint_set_sha256":
        sha256_bytes(canonical(constraint_set)), "uid": os.getuid(), "gid": os.getgid()})
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--make-items", action="store_true")
    p.add_argument("--classes", type=Path, default=HERE / "data-source" / "imagenet_classes.txt")
    p.add_argument("--n-items", type=int, default=150)
    p.add_argument("--split-seed", type=int, default=20261112)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--items", type=Path, default=HERE / "items")
    p.add_argument("--out", type=Path)
    p.add_argument("--eval-seeds", default="11,12")
    p.add_argument("--model-hash")
    p.add_argument("--base-model-hash")
    p.add_argument("--item-set-hash")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--no-aesthetic", action="store_true")
    for k in SAMPLER_DEFAULT:
        p.add_argument(f"--{k.replace('_', '-')}", type=type(SAMPLER_DEFAULT[k]), default=None)
    args = p.parse_args(argv)
    if not args.make_items:
        if args.out is None or (args.checkpoint is None) == (not args.base_only):
            p.error("evaluation needs --out and exactly one of --checkpoint / --base-only")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.make_items:
        return make_items(args.classes, args.n_items, args.split_seed)
    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
