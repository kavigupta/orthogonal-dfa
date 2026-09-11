"""What actually differs between round 0 and round 1?  The generator re-samples the
suffix family every round and rebuilds the DFA from scratch; the ONLY state carried
across rounds is the representative pool (pst.table.prefixes), grown one frame-chain
state per round.  So compare the two pools directly (no oracle):
  - size, length distribution
  - SINK/live composition (by frame sig on the SUPER-string)
  - nesting: is round 0's pool a subset of round 1's? (grow = add, never remove)
  - what did round 1 ADD that round 0 lacked -- its sig/length profile
The pool members are SUPER-strings (alphabet 0-5); sig() applies to those directly.
"""
import pickle, collections
import numpy as np

DUMP = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/mr_dumps/seed2"


def sig(s):
    wc = 0; f0 = f1 = False
    for c in s:
        if c >= 3: wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return "SINK" if (f0 and f1) else (f0, f1, wc % 3)


def coarse(s):
    return "SINK" if sig(s) == "SINK" else "live"


def profile(name, prefixes):
    lens = np.array([len(p) for p in prefixes])
    cs = collections.Counter(coarse(p) for p in prefixes)
    n = len(prefixes)
    print(f"  {name}: {n} prefixes | len min {lens.min()} med {int(np.median(lens))} "
          f"max {lens.max()} | SINK {cs['SINK']} ({cs['SINK']/n*100:.0f}%) "
          f"live {cs['live']} ({cs['live']/n*100:.0f}%)")
    # short core vs full-length population
    short = [p for p in prefixes if len(p) < 30]
    full = [p for p in prefixes if len(p) >= 30]
    print(f"       short-core (<30): {len(short)}  |  full-length (>=30): {len(full)}")
    return cs


def main():
    r0 = pickle.load(open(f"{DUMP}/round_00.pkl", "rb"))
    r1 = pickle.load(open(f"{DUMP}/round_01.pkl", "rb"))
    p0 = [bytes(p) for p in r0["prefixes"]]
    p1 = [bytes(p) for p in r1["prefixes"]]
    print(f"round 0: {r0['n_states']} states, est {r0['true_acc']:.3f}")
    profile("pool0", p0)
    print(f"round 1: {r1['n_states']} states, est {r1['true_acc']:.3f}")
    profile("pool1", p1)

    s0, s1 = set(p0), set(p1)
    inter = s0 & s1
    print(f"\nnesting: |pool0|={len(s0)} |pool1|={len(s1)} "
          f"shared={len(inter)}  pool0\\pool1={len(s0 - s1)}  pool1\\pool0={len(s1 - s0)}")
    print(f"  pool0 subset of pool1? {s0 <= s1}")

    added = [p for p in p1 if p not in s0]
    if added:
        print(f"\nwhat round 1 ADDED ({len(added)} prefixes):")
        lens = np.array([len(p) for p in added])
        cs = collections.Counter(coarse(p) for p in added)
        sigc = collections.Counter(
            "SINK" if sig(p) == "SINK" else f"{'T' if sig(p)[0] else 'F'}{'T' if sig(p)[1] else 'F'}p{sig(p)[2]}"
            for p in added)
        print(f"  len min {lens.min()} med {int(np.median(lens))} max {lens.max()} | "
              f"SINK {cs['SINK']} live {cs['live']}")
        print(f"  by fine sig: {dict(sigc.most_common())}")
    removed = [p for p in p0 if p not in s1]
    if removed:
        lens = np.array([len(p) for p in removed])
        cs = collections.Counter(coarse(p) for p in removed)
        print(f"\nwhat round 1 DROPPED ({len(removed)}): len med {int(np.median(lens))} "
              f"SINK {cs['SINK']} live {cs['live']}")


if __name__ == "__main__":
    main()
