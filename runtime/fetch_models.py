#!/usr/bin/env python
"""Download the ungated model files of area "SD1.5" into plain directories (host step, network on, no GPU).

Each model goes to `models/<name>/` as plain files (no symbolic links, MM-20). huggingface_hub's own download
metadata (`<local_dir>/.cache/huggingface/`) is moved out to `models/hf-meta/<name>/`, so the model directory
holds exactly the repository files at the pinned commit. No token is used or read: `token=False` on every call,
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, and `HF_HOME` points to the fresh `models/hf-home/` (never an existing cache);
the script refuses to run if `HF_TOKEN` is set. Xet transfer is disabled (plain HTTPS, no chunk cache).

After the download, `models/MODELS.json` records per model: repo, revision (40-hex commit), the files and their
SHA-256, and the bytes; `--verify` re-hashes and compares.

Usage (host):
  .venv/bin/python fetch_models.py            # download what is missing, write models/MODELS.json
  .venv/bin/python fetch_models.py --verify   # re-hash every file against models/MODELS.json
  .venv/bin/python fetch_models.py --aesthetic-from <ddpo-pytorch checkout>/ddpo_pytorch/assets/sac+logos+ava1-l14-linearMSE.pth
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"

# name -> (repo, revision, allow_patterns). Revisions resolved 2026-11-12 from the Hub API (`sha` of `main`);
# every repo reported `gated: false`.
SPECS = {
    # The policy base. Only the fp16 safetensors of unet, vae and text encoder plus configs and tokenizer:
    # no safety checker (disabled, as ddpo-pytorch scripts/train.py:100), no .bin, no non-EMA UNet, no full .ckpt.
    "stable-diffusion-v1-5": (
        "stable-diffusion-v1-5/stable-diffusion-v1-5", "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        ["model_index.json", "scheduler/scheduler_config.json", "feature_extractor/preprocessor_config.json",
         "tokenizer/*", "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
         "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
         "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"]),
    # Constraint checker (prompt depicts the animal): SigLIP so400m, a backbone family (siglip) different from the
    # policy's text encoder (OpenAI CLIP ViT-L/14) and from every training reward (README.md "Metrics and tiers").
    "siglip-so400m-patch14-384": (
        "google/siglip-so400m-patch14-384", "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        ["config.json", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json",
         "spiece.model", "tokenizer.json", "tokenizer_config.json"]),
    # Image tower for the LAION aesthetic predictor (descriptive sub-metric; the reward in alternative B).
    "clip-vit-large-patch14": (
        "openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41",
        ["config.json", "model.safetensors", "preprocessor_config.json", "merges.txt", "vocab.json",
         "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def plain_files(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise SystemExit(f"{p} is a symbolic link; the model directory must be plain (MM-20)")
        if p.is_file():
            out[p.relative_to(root).as_posix()] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def download() -> int:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise SystemExit("a Hugging Face token is set in the environment; unset it (these repos are ungated)")
    os.environ["HF_HOME"] = str(MODELS / "hf-home")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    (MODELS / "hf-home").mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    record = {"schema": "ptc_mm.sd15_models", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
              "models": {}}
    for name, (repo, revision, patterns) in SPECS.items():
        dest = MODELS / name
        snapshot_download(repo, revision=revision, local_dir=dest, allow_patterns=patterns, token=False,
                          max_workers=4)
        meta = dest / ".cache"
        if meta.exists():
            target = MODELS / "hf-meta" / name
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(meta), str(target))
        files = plain_files(dest)
        record["models"][name] = {"repo": repo, "revision": revision, "allow_patterns": patterns,
                                  "files": files, "bytes": sum(f["bytes"] for f in files.values())}
        print(f"{name}: {len(files)} files, {record['models'][name]['bytes'] / 1e9:.3f} GB", flush=True)
    (MODELS / "MODELS.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


AESTHETIC_SRC_SHA256 = "sha256:21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0"


def convert_aesthetic(src: Path) -> int:
    """The LAION improved-aesthetic-predictor MLP head (sac+logos+ava1-l14-linearMSE.pth, as shipped in
    ddpo-pytorch 1958463 ddpo_pytorch/assets/) re-saved as safetensors, so no runtime ever unpickles it. The source
    is a plain state dict; it is read once here with `torch.load(weights_only=True)`."""
    if sha256_file(src) != AESTHETIC_SRC_SHA256:
        raise SystemExit(f"{src} is not the pinned aesthetic MLP ({AESTHETIC_SRC_SHA256})")
    import torch
    from safetensors.torch import save_file

    state = torch.load(src, map_location="cpu", weights_only=True)
    dest = MODELS / "aesthetic-mlp"
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "sac+logos+ava1-l14-linearMSE.safetensors"
    save_file({k: v.contiguous() for k, v in state.items()}, str(out))
    source = {"schema": "ptc_mm.sd15_aesthetic_source", "schema_version": "1.0",
              "upstream": "christophschuhmann/improved-aesthetic-predictor (sac+logos+ava1-l14-linearMSE.pth)",
              "copied_from": "kvablack/ddpo-pytorch@1958463f020112c9a7bc85768d296daacc2e1b4b "
                             "ddpo_pytorch/assets/sac+logos+ava1-l14-linearMSE.pth",
              "source_sha256": AESTHETIC_SRC_SHA256, "safetensors_sha256": sha256_file(out),
              "keys": {k: list(v.shape) for k, v in state.items()},
              "clip_backbone": "openai/clip-vit-large-patch14 (models/clip-vit-large-patch14)",
              "training_data": "SAC (Simulacra Aesthetic Captions), LOGOS, AVA (upstream README)"}
    (dest / "SOURCE.json").write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(source, indent=2, sort_keys=True))
    return 0


def verify() -> int:
    record = json.loads((MODELS / "MODELS.json").read_text(encoding="utf-8"))
    bad = 0
    for name, m in record["models"].items():
        got = plain_files(MODELS / name)
        if got != m["files"]:
            bad += 1
            print(f"{name}: MISMATCH", flush=True)
        else:
            print(f"{name}: ok ({len(got)} files)", flush=True)
    return 1 if bad else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--verify", action="store_true")
    p.add_argument("--aesthetic-from", type=Path, help="convert the pinned aesthetic MLP .pth to safetensors")
    args = p.parse_args(argv)
    if args.aesthetic_from is not None:
        return convert_aesthetic(args.aesthetic_from)
    return verify() if args.verify else download()


if __name__ == "__main__":
    sys.exit(main())
