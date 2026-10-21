# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports,protected-access,consider-using-with,redefined-outer-name,cell-var-from-loop,global-statement,too-many-locals,invalid-name,missing-function-docstring,import-error,broad-exception-caught,unused-import,reimported,unused-variable,line-too-long
"""Pickle a round's state right before its counterexample search, with the oracle
left out, so the search can be replayed without redoing the family search."""

import pickle

from orthogonal_dfa.superlanguage.oracle import LiftedOracle


class _Dumper(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, LiftedOracle):
            return ("lifted", obj._seed)
        return None


class _Loader(pickle.Unpickler):
    def __init__(self, f, base, vocab):
        super().__init__(f)
        self.base, self.vocab = base, vocab

    def persistent_load(self, pid):
        kind, seed = pid
        assert kind == "lifted"
        return LiftedOracle(self.base, self.vocab, seed=seed)


def dump(path, **objs):
    with open(path, "wb") as f:
        _Dumper(f, protocol=pickle.HIGHEST_PROTOCOL).dump(objs)


def load(path, base, vocab):
    with open(path, "rb") as f:
        return _Loader(f, base, vocab).load()
