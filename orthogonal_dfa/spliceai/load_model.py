import os
import pickle

import torch

MODULE_RENAME_MAP = {"spliceai_torch": "orthogonal_dfa.spliceai.module"}


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


def load_spliceai(size, seed):
    assert size in (400, 10000, "10k")
    if size == 10000:
        size = "10k"
    return (
        torch.load(
            os.path.join("data/pretrained_models", f"spliceai-{size}-{seed}.pt"),
            weights_only=False,
            pickle_module=remapping_pickle(),
        )
        .eval()
        .cuda()
    )
