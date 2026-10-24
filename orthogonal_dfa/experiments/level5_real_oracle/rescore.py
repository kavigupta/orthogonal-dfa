# pylint: disable=wrong-import-position,wrong-import-order,protected-access,consider-using-with,invalid-name,missing-function-docstring,import-error,line-too-long
"""Score every round's DFA of finished runs on a large fresh sample.

  rescore.py OUT... [--n-half N]

Each OUT is a harness output directory with round<n>.pkl and summary.json.  phi is
fit on one half and read on the other, averaged over both directions.  Appends one
record per run to rescore.jsonl in that run's directory.
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fullloop_mem import VOCAB, score

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual_len40 import (
    gate_residual_oracle,
)
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler

ap = argparse.ArgumentParser()
ap.add_argument("outs", nargs="+")
ap.add_argument("--n-half", type=int, default=30000)
a = ap.parse_args()

base = gate_residual_oracle(
    default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85
)
for out in a.outs:
    seed = json.load(open(os.path.join(out, "summary.json")))["seed"]
    oracle = LiftedOracle(base, VOCAB, seed=seed)
    halves = []
    for sd in (7777, 8888):
        rng = np.random.default_rng(sd)
        samp = SuperSampler(VOCAB, 36)
        s = [samp.sample(rng, VOCAB.alphabet_size) for _ in range(a.n_half)]
        halves.append((s, np.asarray(oracle.membership_queries(s), dtype=float)))
    rec = {"out": out, "seed": seed, "n_half": a.n_half, "rounds": {}}
    for path in sorted(glob.glob(os.path.join(out, "round*.pkl"))):
        n = int(re.search(r"round(\d+)\.pkl$", path).group(1))
        dfa = pickle.load(open(path, "rb"))
        ab = score(dfa, halves)[0]
        ba = score(dfa, halves[::-1])[0]
        rec["rounds"][n] = dict(phi=(ab + ba) / 2, ab=ab, ba=ba, states=len(dfa.states))
    with open(os.path.join(out, "rescore.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
