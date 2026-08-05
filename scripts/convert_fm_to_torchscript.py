"""Convert the fixed-motif (FM) model to self-contained TorchScript artifacts.

    python scripts/convert_fm_to_torchscript.py           # seeds 1..5
    python scripts/convert_fm_to_torchscript.py 1 3       # specific seeds
"""

import os
import sys

import torch

from orthogonal_dfa.spliceai.load_model import PRETRAINED_DIR

FM_REPO = "/mnt/md0/ExpeditionsCommon/spliceai/Canonical"
FM_MODEL_PREFIX = f"{FM_REPO}/model/msp-273.665a3"
# Any width > cl (400); the trace generalizes across lengths (conv/attention crop off
# tensor shapes).
TRACE_WIDTH = 560


def _load_eager(seed):
    if FM_REPO not in sys.path:
        sys.path.insert(0, FM_REPO)
    # modular_splicing lives in FM_REPO (on sys.path above), not in this package.
    from modular_splicing.utils.io import load_model  # pylint: disable=import-error

    _, model = load_model(f"{FM_MODEL_PREFIX}_{seed}")  # picks the latest step
    return model.eval().cuda()


def convert(seed):
    model = _load_eager(seed)
    example = torch.zeros(2, TRACE_WIDTH, 4, device="cuda")
    example[:, torch.arange(TRACE_WIDTH), torch.randint(0, 4, (TRACE_WIDTH,))] = 1.0
    with torch.no_grad():
        traced = torch.jit.trace(model, (example,), check_trace=True)
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    path = os.path.join(PRETRAINED_DIR, f"fm-{seed}.traced.pt")
    torch.jit.save(traced, path)
    return path


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3, 4, 5]
    for seed in seeds:
        path = convert(seed)
        print(f"seed {seed}: {path} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
