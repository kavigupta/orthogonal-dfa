"""Each seed's first residual oracle on the eval strings: its accept rate, its phi with
the stage-1 verdict it should be blind to, and the features it still carries.

    python -m orthogonal_dfa.experiments.raw_acgt.sanity"""

import numpy as np

from ..length_oracles.residual_report import minimize
from .common import (
    EVAL,
    SEEDS,
    H,
    ResidualRaw,
    accepting,
    corr,
    end_states,
    features,
    kept_stage_one,
    load_flat,
    saved_centred,
    top_features,
    verdict,
)


def main():
    flat = load_flat()
    c_fit, c_ev = saved_centred("fit"), saved_centred("eval")
    y = c_ev > 0
    direct = flat.membership_queries([r.tobytes() for r in EVAL[:2000]])
    print(
        f"flat accept {y.mean():.3f}; agrees with the oracle {(direct == y[:2000]).mean()}"
    )
    feats = features(EVAL)
    for s in SEEDS:
        st = kept_stage_one(s)["dfa"]
        p1 = minimize(st, accepting(end_states(st, EVAL), y))
        res = ResidualRaw(flat, p1, c_fit)
        q = end_states(p1, EVAL)
        r = c_ev > res.offsets[q]
        direct = res.membership_queries([x.tobytes() for x in EVAL[:2000]])
        rates = [
            round(float(r[q == k].mean()), 3) for k in range(len(p1["transitions"]))
        ]
        print(
            f"s{s}: stage 1 {len(st['transitions'])} st, minimal "
            f"{len(p1['transitions'])}; residual accept {r.mean():.3f} (per state "
            f"{rates}); phi with stage 1 {corr(verdict(q, y)[H:], r[H:]):+.3f}; with "
            f"flat {corr(r, y):+.3f}; agrees with the oracle {np.mean(direct == r[:2000])}"
        )
        print(
            "   residual vs:",
            ", ".join(f"{n} {v:+.3f}" for n, v in top_features(r, feats, 8)),
        )


if __name__ == "__main__":
    main()
