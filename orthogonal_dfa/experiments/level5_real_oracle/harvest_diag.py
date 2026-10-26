# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports,protected-access,consider-using-with,redefined-outer-name,cell-var-from-loop,global-statement,too-many-locals,invalid-name,missing-function-docstring,import-error,broad-exception-caught,unused-import,reimported,unused-variable,line-too-long
"""fullloop_mem.py with every round ending and family FNR logged to <out>/diag.jsonl."""

import faulthandler
import json
import os
import signal
import sys

faulthandler.register(signal.SIGUSR1, all_threads=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orthogonal_dfa.l_star import cluster
from orthogonal_dfa.l_star import counterexample_search as cxs

OUT = sys.argv[sys.argv.index("--out") + 1]


def log(rec):
    with open(f"{OUT}/diag.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


orig_judge = cluster.judge_family


def judge_family(pst, gate, v, vs, family_size):
    j = orig_judge(pst, gate, v, vs, family_size)
    log(
        dict(
            kind="family",
            fnr=j.fnr,
            reason=j.reason,
            size=len(j.vs),
            family_size=family_size,
            fnr_limit=pst.fnr_limit,
        )
    )
    return j


cluster.judge_family = judge_family

orig_end = cxs.ending_test


def counts_json(counts):
    """Each count list as (total, distinct); any other field as is."""
    out = {}
    for name, v in vars(counts).items():
        if isinstance(v, list):
            out[name] = (len(v), len(set(map(bytes, v))))
        elif isinstance(v, dict):
            flat = [z for zs in v.values() if isinstance(zs, list) for z in zs]
            out[name] = (len(flat), len(set(map(bytes, flat))))
        else:
            out[name] = v
    return out


def ending_test(counts, *args, **kwargs):
    e = orig_end(counts, *args, **kwargs)
    if e is not None:
        log(
            dict(
                kind="ending",
                ending=e.kind,
                harvest_len=len(e.harvest),
                distinct_harvest=len(set(map(bytes, e.harvest))),
                counts=counts_json(counts),
                kwargs=kwargs,
            )
        )
    return e


cxs.ending_test = ending_test

import traceback

import family_ckpt

from orthogonal_dfa.l_star import counterexample_synthesis as cs

orig_search = cs._counterexample_search
families = []


def counterexample_search(pst, vs, *, acc_threshold, selection):
    # Restarts search one family several times; save each family once.
    if not families or families[-1] is not vs:
        families.append(vs)
        name = f"{OUT}/family_r{len(families) - 1}.pkl"
        try:
            family_ckpt.dump(name, pst=pst, vs=vs, acc_threshold=acc_threshold)
            log(dict(kind="dumped", round=len(families) - 1))
        except Exception:
            log(dict(kind="dump_failed", error=traceback.format_exc()))
    return orig_search(pst, vs, acc_threshold=acc_threshold, selection=selection)


cs._counterexample_search = counterexample_search

import fullloop_mem

fullloop_mem.main()
