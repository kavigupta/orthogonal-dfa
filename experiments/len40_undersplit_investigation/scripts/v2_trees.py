"""Localize v2 seed-1's round-0 accept-all collapse: compare the decision trees and
DFAs of the three v2 seeds' round 0.  Family is normal (measured), so the collapse
is downstream -- in the tree or the DFA induction from it.  Is s1's tree degenerate
(few leaves / everything on accept side / shallow), or is the tree fine but the DFA
labeling wrong?"""
import pickle
NAME = {0: "TAG", 1: "GGTA", 2: "TGA", 3: "TAA", 4: "AGGT", 5: "X", 6: "Y"}
D = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad"


def named(seq):
    return "".join("[" + NAME.get(int(s), str(s)) + "]" for s in seq) or "eps"


def render_midfix(m):
    flat = m[0] if isinstance(m, (tuple, list)) and m and isinstance(m[0], (tuple, list)) else m
    return named(flat)


for s in (0, 1, 2):
    try:
        d = pickle.load(open(f"{D}/rounds_v2_s{s}/round_00.pkl", "rb"))
    except FileNotFoundError:
        print(f"v2_s{s}: missing"); continue
    dfa, dt = d["dfa"], d["dt"]
    acc = sorted(dfa.final_states); rej = sorted(set(dfa.states) - set(dfa.final_states))
    leaves = list(dt.leaves())
    acc_leaves = dt.accepting_leaves()
    print(f"\n===== v2_s{s} round 0  (est {d['est']:.3f}, {d['num_states']} DFA states) =====")
    print(f"  DFA: {len(dfa.states)} states | accepting {acc} | rejecting {rej}")
    print(f"  tree: {dt.num_states} leaves, depth {dt.depth} | "
          f"accepting leaves {len(acc_leaves)}/{len(leaves)}")
    # is the tree degenerate? show it (small) or summarize (large)
    lines = dt.render(render_midfix, indent=4)
    if len(lines) <= 40:
        print("  tree:")
        for ln in lines:
            print("  " + ln)
    else:
        print(f"  tree has {len(lines)} lines (large); top splits:")
        for ln in lines[:14]:
            print("  " + ln)
