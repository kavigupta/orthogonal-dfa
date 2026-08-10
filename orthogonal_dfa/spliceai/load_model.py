import hashlib
import os
import pickle

import torch
from torch import nn

MODULE_RENAME_MAP = {
    "spliceai_torch": "orthogonal_dfa.spliceai.module",
    "splice_point_identifier": "orthogonal_dfa.spliceai.lssi",
}


class renamed_symbol_unpickler(pickle.Unpickler):
    """
    Unpicler that renames modules and symbols as specified in the
    MODULE_RENAME_MAP and SYMBOL_RENAME_MAP dictionaries.
    """

    def find_class(self, module, name):
        module = MODULE_RENAME_MAP.get(module, module)

        try:
            return super().find_class(module, name)
        except:
            print("Could not find", (module, name))
            raise


class remapping_pickle:
    """
    An instance of this class will behave like the pickle module, but
    will use the renamed_symbol_unpickler class instead of the default
    Unpickler class.
    """

    def __getattribute__(self, name):
        if name == "Unpickler":
            return renamed_symbol_unpickler
        return getattr(pickle, name)

    def __hasattr__(self, name):
        return hasattr(pickle, name)


PRETRAINED_DIR = "data/pretrained_models"


def load_spliceai(size, seed):
    assert size in (400, 10000, "10k")
    if size == 10000:
        size = "10k"
    return (
        torch.load(
            os.path.join(PRETRAINED_DIR, f"spliceai-{size}-{seed}.pt"),
            weights_only=False,
            pickle_module=remapping_pickle(),
        )
        .eval()
        .cuda()
    )


def load_lssi(which, seed):
    assert which in ("donor", "acceptor")
    return (
        torch.load(
            os.path.join(PRETRAINED_DIR, f"{which}-{seed}.pt"),
            weights_only=False,
            pickle_module=remapping_pickle(),
        )
        .eval()
        .cuda()
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class TracedFM(nn.Module):
    """The FM TorchScript trace, tagged with the hash of the file it came from.

    permacache's ``stable_hash`` walks an eager model's state dict, but a
    ScriptModule is opaque to it and raises instead, so anything permacached on
    the model (``median_threshold``, the E-L* oracle) cannot compute a key.
    ``__permacache_hash__`` answers with the artifact's hash, which identifies
    the weights just as precisely: regenerating a trace changes the file.
    """

    def __init__(self, model, trace_sha256: str):
        super().__init__()
        self.model = model
        self.trace_sha256 = trace_sha256

    def forward(self, x):
        return self.model(x)

    def __permacache_hash__(self):
        return {"fm_trace_sha256": self.trace_sha256}


def load_fm(seed=1):
    """
    Load the fixed-motif (FM) model as a self-contained TorchScript artifact.

    Reads the same acceptor/donor logits as SpliceAI, so ``SpliceAIExonScore``
    wraps it identically.

    The trace has ``cuda:0`` baked into every device it names, so select a GPU
    with ``CUDA_VISIBLE_DEVICES`` rather than moving the module: ``.to("cuda:1")``
    moves the parameters while the baked constants keep producing cuda:0 tensors.
    """
    path = os.path.join(PRETRAINED_DIR, f"fm-{seed}.traced.pt")
    assert os.path.exists(path), (
        f"{path} is missing; generate the FM traces (on the machine with the "
        f"modular_splicing repo) via scripts/convert_fm_to_torchscript.py"
    )
    return TracedFM(torch.jit.load(path), file_sha256(path)).eval().cuda()
