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


def ending_test(counts, config, depth, leaves):
    e = orig_end(counts, config, depth, leaves)
    if e is not None:
        edges = [
            dict(
                edge=str(k),
                leaf=k[0] in leaves,
                traversals=n,
                undecided=len(counts.edge_harvest.get(k, [])),
            )
            for k, n in counts.traversals.items()
        ]
        tot_t = sum(x["traversals"] for x in edges)
        tot_u = sum(x["undecided"] for x in edges)
        log(
            dict(
                kind="ending",
                ending=e.kind,
                edge=str(e.edge),
                probes=counts.probes,
                depth=depth,
                harvest_len=len(e.harvest),
                start_harvest=len(counts.start_harvest),
                searches=counts.searches,
                middles=len(counts.middles),
                search_queries=counts.search_queries,
                edge_rate=config.edge_rate,
                edge_excess=config.edge_excess,
                edge_traffic=config.edge_traffic,
                start_rate=config.start_rate(depth),
                overall_undecided_per_traversal=tot_u / max(tot_t, 1),
                distinct_harvest=len(set(map(repr, e.harvest))),
                edges=edges,
            )
        )
    return e


cxs.ending_test = ending_test

import traceback

import family_ckpt

from orthogonal_dfa.l_star import counterexample_synthesis as cs

orig_search = cs._counterexample_search
calls = [0]


def counterexample_search(pst, vs, *, acc_threshold):
    try:
        family_ckpt.dump(
            f"{OUT}/family_r{calls[0]}.pkl", pst=pst, vs=vs, acc_threshold=acc_threshold
        )
        log(dict(kind="dumped", round=calls[0]))
    except Exception:
        log(dict(kind="dump_failed", error=traceback.format_exc()))
    calls[0] += 1
    return orig_search(pst, vs, acc_threshold=acc_threshold)


cs._counterexample_search = counterexample_search

import fullloop_mem

fullloop_mem.main()
