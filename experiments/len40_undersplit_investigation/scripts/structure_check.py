"""Did the learner find the right STRUCTURE (partition), independent of accept/reject
labels?  phi_frame01 in the dumps is the LABELED output and is dragged down by wrong
labels (round 1 over-rejects, accept_rate ~0.42).  Here, per seed/round dump, we walk the
DFA and compute:
  - raw phi_frame01 (from the DFA's own final_states) -- should match the dump
  - STRUCTURE-optimal phi_frame01: relabel each state by the majority frame01 of the
    strings that walk there (the best any labeling of THIS partition can do)
  - SINK/live homogeneity of the walk partition
  - #states, #reachable states
If structure-optimal phi ~ +1.0 but raw is low, the partition is right and only the labels
are wrong (denoise territory).  No oracle needed.
"""
import pickle, glob, os, collections, math
import numpy as np
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

BASE = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad"
DIRS = [("seed0", f"{BASE}/mr_dumps/seed0"), ("seed1", f"{BASE}/mr_multi/seed1"),
        ("seed2", f"{BASE}/mr_dumps/seed2"), ("seed3", f"{BASE}/mr_multi/seed3")]
NEVAL = 4000


def sig(s):
    wc = 0; f0 = f1 = False
    for c in s:
        if c >= 3: wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return "SINK" if (f0 and f1) else "live"


def entropy(cs):
    n = sum(cs)
    return -sum((c / n) * math.log(c / n) for c in cs if c > 0) if n else 0.0


def homogeneity(states, labels):
    by = collections.defaultdict(collections.Counter); tot = collections.Counter(); lt = collections.Counter()
    for s, g in zip(states, labels):
        by[s][g] += 1; tot[s] += 1; lt[g] += 1
    H = entropy(list(lt.values())); n = len(states)
    Hc = sum((tot[s] / n) * entropy(list(sub.values())) for s, sub in by.items())
    return 1 - Hc / H if H else 1.0


def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def walk(dfa, w):
    s = dfa.initial_state
    for c in w:
        s = dfa.transitions[s][c]
    return s


def main():
    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    print(f"{'seed':6} {'rnd':3} {'states':6} {'reach':5} {'raw_phi':8} {'STRUCT_phi':10} "
          f"{'SINK/live_homog':14} {'#accept_states':13}")
    for name, d in DIRS:
        for f in sorted(glob.glob(f"{d}/round_*.pkl")):
            rec = pickle.load(open(f, "rb"))
            seed = rec["seed"]; rd = os.path.basename(f).split("_")[1].split(".")[0]
            dfa = rec["dfa"]
            samp = SuperSampler(vocab, 36); rng = np.random.default_rng(seed + 999)
            supers = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(NEVAL)]
            f01 = np.array([0 if sig(w) == "SINK" else 1 for w in supers])  # accept iff live
            st = [walk(dfa, w) for w in supers]
            acc = getattr(dfa, "final_states", set())
            raw_call = np.array([1 if s in acc else 0 for s in st])
            # structure-optimal relabel: state -> majority f01 of its members
            by = collections.defaultdict(list)
            for s, y in zip(st, f01):
                by[s].append(y)
            relabel = {s: (1 if np.mean(v) >= 0.5 else 0) for s, v in by.items()}
            opt_call = np.array([relabel[s] for s in st])
            reach = len(set(st))
            print(f"{name:6} {rd:3} {rec['n_states']:6d} {reach:5d} {phi(raw_call, f01):+.3f}   "
                  f"{phi(opt_call, f01):+.3f}      {homogeneity(st, list(f01)):.3f}          "
                  f"{sum(1 for s in acc if s in by)}")


if __name__ == "__main__":
    main()
