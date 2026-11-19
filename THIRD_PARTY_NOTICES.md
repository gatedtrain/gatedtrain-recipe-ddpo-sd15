# Third-party notices

This repository is licensed under the Apache License 2.0 (`LICENSE`). It contains code ported from, and data copied
from, the projects below: kvablack/ddpo-pytorch (MIT License) and, through it, huggingface/diffusers (Apache License
2.0). Model weights are not part of this repository; their
licenses are listed in `README.md` ("Data and licenses").

## kvablack/ddpo-pytorch

- Source: https://github.com/kvablack/ddpo-pytorch, commit `1958463f020112c9a7bc85768d296daacc2e1b4b`
- License: MIT (the `LICENSE` file of that commit, reproduced below)
- Used in this repository:
  - `runtime/train_ddpo.py`: a port of the DDIM step with log-probability, the sampling loop with classifier-free
    guidance, the PPO update, `PerPromptStatTracker`, the JPEG-compressibility reward and the hyperparameters onto
    current diffusers and peft; every ported piece is cited by file and line of that commit in the script.
  - `runtime/eval_images.py`: the shared sampler block (the same ported DDIM step and sampling loop) and the
    JPEG-compressibility measurement.
  - `runtime/data-source/imagenet_classes.txt`: a byte copy of `ddpo_pytorch/assets/imagenet_classes.txt` at that
    commit (SHA-256 `a40d9be82eb4738becd1dd69991b89f87f309975e860a6f3f45395d8c06a7e0d`), the ImageNet (ILSVRC-2012)
    class names; `runtime/prompts/train.jsonl` and `runtime/items/items.jsonl` are derived from it.
  - `runtime/fetch_models.py --aesthetic-from` converts the aesthetic-predictor head shipped in that repository
    (`ddpo_pytorch/assets/sac+logos+ava1-l14-linearMSE.pth`, originally from
    christophschuhmann/improved-aesthetic-predictor, Apache-2.0); the weights are not committed here.

```
MIT License

Copyright (c) 2023 Kevin Black

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## huggingface/diffusers (Apache-2.0)

- Source: https://github.com/huggingface/diffusers, commit `fc6acb6b97e93d58cb22b5fee52d884d77ce84d8`, the commit
  named by ddpo-pytorch's copies
- Files: ddpo-pytorch's `ddpo_pytorch/diffusers_patch/ddim_with_logprob.py` is copied, with modifications, from
  `src/diffusers/schedulers/scheduling_ddim.py`, and `ddpo_pytorch/diffusers_patch/pipeline_with_logprob.py` from
  `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py` at that commit (each file's first line says
  so). The DDIM step with log-probability and the sampling loop with classifier-free guidance in
  `runtime/train_ddpo.py` and `runtime/eval_images.py` (the shared sampler block) are ported from those two copies.
- Copyright lines of the diffusers files at that commit:
  - `scheduling_ddim.py`: "Copyright 2023 Stanford University Team and The HuggingFace Team. All rights reserved."
  - `pipeline_stable_diffusion.py`: "Copyright 2023 The HuggingFace Team. All rights reserved."
- License: Licensed under the Apache License, Version 2.0 (the same text as `LICENSE`). diffusers has no NOTICE file.
