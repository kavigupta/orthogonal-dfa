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
        if module in MODULE_RENAME_MAP:
            module = MODULE_RENAME_MAP[module]

        try:
            return super(renamed_symbol_unpickler, self).find_class(module, name)
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


def load_spliceai(path):
    return torch.load(
        os.path.join(path, max(os.listdir(path), key=int)),
        weights_only=False,
        pickle_module=remapping_pickle(),
    )
