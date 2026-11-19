"""Each kept stage-1 DFA's states: transitions on A, C, G, T, share of the eval strings
ending there, and the flat oracle's accept rate on them.

    python -m orthogonal_dfa.experiments.raw_acgt.states"""

from .common import EVAL, SEEDS, accepting, end_states, kept_stage_one, saved_centred


def main():
    y = saved_centred("eval") > 0
    for s in SEEDS:
        d = kept_stage_one(s)["dfa"]
        q = end_states(d, EVAL)
        own = [i for i, a in enumerate(d["accepting"]) if a]
        print(
            f"s{s} start {d['start']} own-accepting {own} "
            f"fit-accepting {sorted(accepting(q, y))}"
        )
        for st, row in enumerate(d["transitions"]):
            m = q == st
            rate = f"{y[m].mean():.3f}" if m.any() else "nan"
            print(f"  q{st:2d} -> {row}  share {m.mean():.3f}  oracle-accept {rate}")


if __name__ == "__main__":
    main()
