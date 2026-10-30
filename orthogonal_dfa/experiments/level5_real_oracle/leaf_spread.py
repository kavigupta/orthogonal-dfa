# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports,protected-access,consider-using-with,redefined-outer-name,cell-var-from-loop,global-statement,too-many-locals,invalid-name,missing-function-docstring,import-error,broad-exception-caught,unused-import,reimported,unused-variable,line-too-long
"""Where pool prefixes, and prefixes of held-out strings by length, sift in a snapshot's tree."""

import collections
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import family_ckpt
from fullloop_mem import VOCAB

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import (
    gate_residual_oracle,
)
from orthogonal_dfa.l_star.hypothesis_dfa import HypothesisDFA
from orthogonal_dfa.l_star.suffix_family import SuffixFamily
from orthogonal_dfa.spliceai.load_model import load_spliceai

ckpt, snap = sys.argv[1], sys.argv[2]
base = gate_residual_oracle(
    default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85
)
d = family_ckpt.load(ckpt, base, VOCAB)
pst, vs = d["pst"], d["vs"]
fam = SuffixFamily(pst, vs)
s = pickle.load(open(snap, "rb"))
h = HypothesisDFA(lambda z: fam.is_accept(z, b""), s["tree"])
pool = [p for p, keep in zip(pst.table.prefixes, pst.table.representative) if keep]
halves = pickle.load(
    open(os.path.join(os.path.dirname(os.path.abspath(ckpt)), "halves.pkl"), "rb")
)
held = halves[0][0][:150]
out = {"pool": collections.Counter(str(h.sift(p)) for p in pool).most_common(8)}
for L in (1, 2, 4, 8, 16, 36):
    out[f"held_len{L}"] = collections.Counter(
        str(h.sift(x[:L])) for x in held
    ).most_common(6)
print(json.dumps(out, indent=0))
