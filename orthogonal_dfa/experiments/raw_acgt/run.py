"""Stage 1 learns the flat oracle over length-40 ACGT strings; stage k+1 learns it with
the minimized product of stages 1..k taken out (see ResidualRaw).  Each stage is round 0
with 4 restarts.  A seed stops early once a stage's residual phi is under .02 or the
product's phi gains under .003.

    python -u -m orthogonal_dfa.experiments.raw_acgt.run --seed S

Writes data/neural/raw_acgt/runs/sS/ for stage 1, and residual/sS/stages.jsonl plus
stageK/restarts.json for the rest."""

import argparse
import json
import os
import time

import numpy as np

from orthogonal_dfa.l_star.learn import NUM_PREFIXES
from orthogonal_dfa.l_star.prefix_suffix_tracker import (
    PrefixSuffixTracker,
    SearchConfig,
)
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.statistics import compute_suffix_size_counterexample_gen

from ..length_oracles import run as length_run
from ..length_oracles.residual_report import minimize, product
from .common import (
    EVAL,
    FRAMES,
    LENGTH,
    RESIDUAL,
    RUNS,
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


def search_config():
    """``build_pst``'s config at signal 0.05."""
    return SearchConfig(
        suffix_size_counterexample_gen=compute_suffix_size_counterexample_gen(
            0.01, 0.55
        ),
        min_signal_strength=0.05,
        num_addtl_prefixes=NUM_PREFIXES,
        min_suffix_frequency=0.02,
        require_accept_preserving=True,
        cross_limit=1.5e-7,
        acceptable_fnr=0.01,
        max_coverage_error=1 / 3,
    )


def synthesize(oracle, seed, out):
    """The restarts of one round-0 run against ``oracle``, recorded in ``out``."""
    # build_pst seeds the learner with 0 whatever its seed, which a deterministic
    # oracle turns into one run for every seed.
    pst = PrefixSuffixTracker.create(
        UniformSampler(LENGTH),
        np.random.default_rng(seed),
        oracle,
        search_config(),
        num_prefixes=NUM_PREFIXES,
    )
    return length_run.round_zero(pst, out)


def stage_one(seed, flat):
    out = RUNS / f"s{seed}"
    os.makedirs(out, exist_ok=True)
    started = time.time()
    restarts = synthesize(flat, seed, out)
    status = length_run.git("status", "--porcelain", "--untracked-files=no")
    with open(out / "run.json", "w") as f:
        json.dump(
            dict(
                oracle="flat-raw",
                length=LENGTH,
                seed=seed,
                commit=length_run.git("rev-parse", "--short=8", "HEAD"),
                dirty=None if status is None else bool(status),
                hours=(time.time() - started) / 3600,
                restarts=len(restarts),
                family_size=restarts[0]["family_size"] if restarts else None,
                family_fnr=restarts[0]["family_fnr"] if restarts else None,
            ),
            f,
            indent=1,
        )


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


def residual_stages(seed, flat, last):
    out = RESIDUAL / f"s{seed}"
    os.makedirs(out, exist_ok=True)
    c_fit, c_ev = saved_centred("fit"), saved_centred("eval")
    y = c_ev > 0
    feats = features(EVAL)
    stage1 = kept_stage_one(seed)
    prod = minimize(stage1["dfa"], accepting(end_states(stage1["dfa"], EVAL), y))
    prod_phi = phi(end_states(prod, EVAL), y)
    with open(out / "stages.jsonl", "w") as log:
        log.write(
            json.dumps(
                dict(
                    seed=seed,
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
        for k in range(2, last + 1):
            started = time.time()
            res = ResidualRaw(flat, prod, c_fit)
            r_ev = c_ev > res.offsets[end_states(prod, EVAL)]
            os.makedirs(out / f"stage{k}", exist_ok=True)
            restarts = synthesize(res, seed, out / f"stage{k}")
            rec = stage_record(r_ev, y, prod, restarts, feats)
            rec = dict(seed=seed, stage=k, hours=(time.time() - started) / 3600, **rec)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--last", type=int, default=4)
    a = ap.parse_args()
    flat = load_flat()
    stage_one(a.seed, flat)
    residual_stages(a.seed, flat, a.last)


if __name__ == "__main__":
    main()
