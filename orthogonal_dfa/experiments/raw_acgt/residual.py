"""Residual stages 2-4 for one seed: stage k+1 learns the flat oracle with the minimized
product of stages 1..k taken out (see ResidualRaw).  A seed stops early once a stage's
residual phi is under .02 or the product's phi gains under .003.

    python -u -m orthogonal_dfa.experiments.raw_acgt.residual --seed S

Appends one line per stage to data/neural/raw_acgt/residual/sS/stages.jsonl, and writes
each stage's restarts to stageK/restarts.json."""

import argparse
import json
import os
import time

from ..length_oracles.residual_report import minimize, product
from .common import (
    EVAL,
    FRAMES,
    RESIDUAL,
    H,
    ResidualRaw,
    accepting,
    corr,
    end_states,
    features,
    kept_stage_one,
    load_flat,
    phi,
    saved_centred,
    top_features,
    verdict,
)
from .run import synthesize


def stage_record(r_ev, y, prod, restarts, feats):
    kept = max(restarts, key=lambda r: r["pick_score"])
    st = kept["dfa"]
    qs = end_states(st, EVAL)
    v = verdict(qs, r_ev)
    pairs = product(prod, st)
    cells = end_states(pairs, EVAL)
    new = minimize(pairs, accepting(cells, y))
    pv = verdict(end_states(new, EVAL), y)
    return dict(
        restarts=[
            dict(
                restart=r["restart"],
                pick=r["pick_score"],
                ending=r["ending"],
                states=len(r["dfa"]["transitions"]),
                residual_phi=phi(end_states(r["dfa"], EVAL), r_ev),
            )
            for r in restarts
        ],
        restart=kept["restart"],
        dfa=st,
        stage_states=len(st["transitions"]),
        residual_accept=float(r_ev.mean()),
        residual_phi=phi(qs, r_ev),
        stage_flat_phi=phi(qs, y),
        stage_features=top_features(v, feats),
        stage_frames={n: corr(v[H:], feats[n][H:]) for n in FRAMES},
        reachable=len(pairs["transitions"]),
        product=new,
        product_states=len(new["transitions"]),
        product_phi=phi(cells, y),
        product_features=top_features(pv, feats),
        product_frames={
            n: corr(pv[H:], feats[n][H:]) for n in ("f0 open", "f0|1 open")
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--last", type=int, default=4)
    a = ap.parse_args()
    out = RESIDUAL / f"s{a.seed}"
    os.makedirs(out, exist_ok=True)
    flat = load_flat()
    c_fit, c_ev = saved_centred("fit"), saved_centred("eval")
    y = c_ev > 0
    feats = features(EVAL)
    stage1 = kept_stage_one(a.seed)
    prod = minimize(stage1["dfa"], accepting(end_states(stage1["dfa"], EVAL), y))
    prod_phi = phi(end_states(prod, EVAL), y)
    with open(out / "stages.jsonl", "a") as log:
        log.write(
            json.dumps(
                dict(
                    seed=a.seed,
                    stage=1,
                    restart=stage1["restart"],
                    dfa=stage1["dfa"],
                    stage_states=len(stage1["dfa"]["transitions"]),
                    product=prod,
                    product_states=len(prod["transitions"]),
                    product_phi=prod_phi,
                )
            )
            + "\n"
        )
        log.flush()
        for k in range(2, a.last + 1):
            started = time.time()
            res = ResidualRaw(flat, prod, c_fit)
            r_ev = c_ev > res.offsets[end_states(prod, EVAL)]
            os.makedirs(out / f"stage{k}", exist_ok=True)
            restarts = synthesize(res, a.seed, out / f"stage{k}")
            rec = stage_record(r_ev, y, prod, restarts, feats)
            rec = dict(
                seed=a.seed, stage=k, hours=(time.time() - started) / 3600, **rec
            )
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(
                f"stage {k}: residual phi {rec['residual_phi']:.3f}; product "
                f"{rec['product_states']} states phi {rec['product_phi']:.3f}",
                flush=True,
            )
            if rec["residual_phi"] < 0.02 or rec["product_phi"] < prod_phi + 0.003:
                break
            prod, prod_phi = rec["product"], rec["product_phi"]


if __name__ == "__main__":
    main()
