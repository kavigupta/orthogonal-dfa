"""The features the flat oracle and each kept stage-1 DFA's verdict track most: base
counts, and the base at each of the first and last 3 positions.

    python -m orthogonal_dfa.experiments.raw_acgt.features"""

from .common import (
    EVAL,
    FRAMES,
    SEEDS,
    end_states,
    features,
    kept_stage_one,
    saved_centred,
    top_features,
    verdict,
)


def show(pairs):
    return ", ".join(f"{k} {v:+.2f}" for k, v in pairs)


def main():
    y = saved_centred("eval") > 0
    feats = {k: v for k, v in features(EVAL).items() if k not in FRAMES}
    print("oracle:", show(top_features(y, feats, 8)))
    for s in SEEDS:
        r = kept_stage_one(s)
        v = verdict(end_states(r["dfa"], EVAL), y)
        print(
            f"s{s} r{r['restart']} ({len(r['dfa']['transitions'])} states):",
            show(top_features(v, feats)),
        )


if __name__ == "__main__":
    main()
