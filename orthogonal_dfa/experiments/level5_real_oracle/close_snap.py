# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports,protected-access,consider-using-with,redefined-outer-name,cell-var-from-loop,global-statement,too-many-locals,invalid-name,missing-function-docstring,import-error,broad-exception-caught,unused-import,reimported,unused-variable,line-too-long
"""Close a replay snapshot's hypothesis the way _counterexample_search does at a
round's end, and score it before and after.

  close_snap.py CKPT SNAP.pkl [SNAP.pkl ...]   -> appends to <snap dir>/closed.jsonl
"""

import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import family_ckpt
from fullloop_mem import VOCAB, score

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star import counterexample_synthesis as cs
from orthogonal_dfa.l_star.counterexample_search import CounterexampleSearch
from orthogonal_dfa.l_star.counterexample_synthesis import counterexample_search_config
from orthogonal_dfa.l_star.examples.gate_composition_residual import (
    gate_residual_oracle,
)
from orthogonal_dfa.l_star.suffix_family import SuffixFamily
from orthogonal_dfa.spliceai.load_model import load_spliceai

ckpt, snaps = sys.argv[1], sys.argv[2:]
base = gate_residual_oracle(
    default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85
)
d = family_ckpt.load(ckpt, base, VOCAB)
pst, vs = d["pst"], d["vs"]
halves = pickle.load(
    open(os.path.join(os.path.dirname(os.path.abspath(ckpt)), "halves.pkl"), "rb")
)
family = SuffixFamily(pst, vs)
members = [p for p, keep in zip(pst.table.prefixes, pst.table.representative) if keep]
for snap in snaps:
    t0 = time.time()
    s = pickle.load(open(snap, "rb"))
    search = CounterexampleSearch(
        counterexample_search_config(0.98, 0.01),
        lambda z: family.is_accept(z, b""),
        s["tree"],
        members=members,
    )
    search.hypothesis.edges = dict(s["edges"])
    before = score(cs._hypothesis_dfa(pst, search.hypothesis), halves)
    learned_before = len(search.hypothesis.edges)
    search.close(pst.alphabet_size)
    after = score(cs._hypothesis_dfa(pst, search.hypothesis), halves)
    rec = dict(
        snap=os.path.basename(snap),
        leaves=len(search.hypothesis.leaves),
        edges_before=learned_before,
        edges_after=len(search.hypothesis.edges),
        phi_before=before[0],
        phi_after=after[0],
        top_after=after[1],
        used_after=after[2],
        seconds=time.time() - t0,
    )
    print(json.dumps(rec), flush=True)
    with open(os.path.join(os.path.dirname(snap), "closed.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
