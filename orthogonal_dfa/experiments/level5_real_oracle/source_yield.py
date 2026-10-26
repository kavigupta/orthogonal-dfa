# pylint: disable=wrong-import-position,wrong-import-order,protected-access,consider-using-with,invalid-name,missing-function-docstring,import-error,line-too-long
"""Measure the next round's boundary source: yield of new strings per attempt, and
seconds per attempt, after a round's search replayed from a family checkpoint.

  source_yield.py CKPT [--attempts N] [--restarts K]

Appends one record per attempt to <ckpt dir>/source_yield/attempts.jsonl and the
totals to summary.json.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import family_ckpt
from fullloop_mem import VOCAB

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star import counterexample_synthesis as cs
from orthogonal_dfa.l_star.examples.gate_composition_residual_len40 import (
    gate_residual_oracle,
)
from orthogonal_dfa.l_star.rejection_source import proving_attempts
from orthogonal_dfa.spliceai.load_model import load_spliceai

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--attempts", type=int, default=300)
ap.add_argument("--restarts", type=int, default=1)
a = ap.parse_args()

outdir = os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "source_yield")
os.makedirs(outdir, exist_ok=True)
attempts_path = os.path.join(outdir, "attempts.jsonl")
open(attempts_path, "w").close()

base = gate_residual_oracle(
    default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85
)
d = family_ckpt.load(a.ckpt, base, VOCAB)
pst, vs, acc = d["pst"], d["vs"], d["acc_threshold"]

pre_close = {}
orig_close = cs.CounterexampleSearch.close


def recording_close(self, alphabet_size):
    pre_close["snap"] = self.snapshot()
    orig_close(self, alphabet_size)
    pre_close["closed"] = self.snapshot()


cs.CounterexampleSearch.close = recording_close

t0 = time.time()
pst.config.counterexample_restarts = a.restarts
selection = cs._selection_sample(pst)
search, ending, _ = cs._best_search(
    pst, vs, acc_threshold=acc, index=0, selection=selection
)
search_min = (time.time() - t0) / 60
strings, replay = cs._next_harvest(pst, search, ending, None)
seen0 = set(map(bytes, strings))
print(
    f"search: {ending.kind} in {search_min:.1f} min, harvest {len(strings)} ({len(seen0)} distinct)",
    flush=True,
)

returned = search.snapshot()
variants = {
    "returned": returned,
    "unclosed": pre_close.get("snap", returned),
    "closed": pre_close.get("closed", returned),
}
edges_total = search.hypothesis.tree.num_states * pst.alphabet_size
per_variant = {}
landed = 0
for name, snap in variants.items():
    search.restore(snap)
    seen = set(seen0)
    readers = {
        "stuck": lambda c: c.stuck,
        "ends": lambda c: c.ends,
        "ends+middle": lambda c: c.ends + c.middle,
    }
    reader_seen = {r: set(seen0) for r in readers}
    reader_hits = dict.fromkeys(readers, 0)
    lengths = {"ends": [], "middle": []}
    kinds = {"ends": 0, "stuck": 0, "middle": 0, "by_edge": 0}
    outcomes, new_by_reader, reader_landed = {}, 0, 0
    t_v = time.time()
    for n in range(1, a.attempts + 1):
        t = time.time()
        x = replay.draw()
        outcome, charged = search.probe(x)
        outcomes[
            type(outcome).__name__
            + ("-stuck" if getattr(outcome, "stuck", False) else "")
        ] = (
            outcomes.get(
                type(outcome).__name__
                + ("-stuck" if getattr(outcome, "stuck", False) else ""),
                0,
            )
            + 1
        )
        for kname in ("ends", "stuck", "middle"):
            kinds[kname] += len(getattr(charged, kname))
        kinds["by_edge"] += sum(map(len, charged.undecided_by_edge.values()))
        got = [bytes(z) for z in replay.read(charged)]
        new = [z for z in got if z not in seen]
        seen.update(new)
        reader_landed += bool(new)
        for r, read in readers.items():
            fresh = [bytes(z) for z in read(charged) if bytes(z) not in reader_seen[r]]
            reader_seen[r].update(fresh)
            reader_hits[r] += bool(fresh)
        lengths["ends"] += [len(z) for z in charged.ends]
        lengths["middle"] += [len(z) for z in charged.middle]
        with open(attempts_path, "a") as f:
            f.write(
                json.dumps(
                    dict(
                        variant=name,
                        n=n,
                        s=time.time() - t,
                        outcome=type(outcome).__name__,
                        ends=len(charged.ends),
                        stuck=len(charged.stuck),
                        middle=len(charged.middle),
                        read=len(got),
                        new=len(new),
                    )
                )
                + "\n"
            )
    per_variant[name] = dict(
        edges_learned=len(snap[1]),
        edges_total=edges_total,
        yield_new=reader_landed / a.attempts,
        yield_by_reader={r: h / a.attempts for r, h in reader_hits.items()},
        read_lengths={k: sorted(v) for k, v in lengths.items()},
        charges=kinds,
        outcomes=outcomes,
        seconds_per_attempt=(time.time() - t_v) / a.attempts,
    )
    print(name, json.dumps(per_variant[name]), flush=True)
search.restore(returned)

good = 1 - acc
summary = dict(
    ckpt=a.ckpt,
    ending=ending.kind,
    search_minutes=search_min,
    states=search.hypothesis.tree.num_states,
    attempts=a.attempts,
    variants=per_variant,
    good=good,
    poor=good / 2,
    proving=proving_attempts(good, good / 2),
)
json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=1)
print("SUMMARY", json.dumps(summary), flush=True)
