"""Each stage-1 restart on the eval strings: states reached, held-out phi with the flat
oracle, phi of its own accepting states, and its verdict's phi with the frame predicates.

    python -m orthogonal_dfa.experiments.raw_acgt.analyze"""

import json

import numpy as np

from .common import (
    EVAL,
    FRAMES,
    RUNS,
    SEEDS,
    H,
    corr,
    end_states,
    features,
    saved_centred,
    verdict,
)


def main():
    y = saved_centred("eval") > 0
    feats = features(EVAL)
    print(
        f"oracle accept {y.mean():.3f}",
        " ".join(f"{k} {corr(feats[k][H:], y[H:]):+.3f}" for k in FRAMES),
    )
    for s in SEEDS:
        with open(RUNS / f"s{s}" / "restarts.json") as f:
            restarts = json.load(f)
        kept = max(restarts, key=lambda r: r["pick_score"])["restart"]
        for r in restarts:
            d = r["dfa"]
            q = end_states(d, EVAL)
            v = verdict(q, y)
            own = np.array(d["accepting"], bool)[q]
            frames = " ".join(f"{k} {corr(v[H:], feats[k][H:]):+.2f}" for k in FRAMES)
            print(
                f"s{s} r{r['restart']}{'*' if r['restart'] == kept else ' '} "
                f"{r['ending']:>10s} probes {r['probes']:6d} states "
                f"{len(d['transitions']):3d} reached {len(np.unique(q)):3d} | pick "
                f"{r['pick_score']:+.3f} held-out {corr(v[H:], y[H:]):+.3f} own "
                f"{corr(own[H:], y[H:]):+.3f} accepts {v.mean():.2f} | {frames}"
            )


if __name__ == "__main__":
    main()
