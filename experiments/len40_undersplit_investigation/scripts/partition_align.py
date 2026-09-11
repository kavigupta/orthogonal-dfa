"""Quantify the reframe: round 0's problem is WRONG splits (partition misaligned with
frame-sig structure), not too MANY splits.  For each round, walk a fixed eval set,
get the produced-state partition, and score it against the target frame-sig partition
with homogeneity / completeness / V-measure (no oracle needed).  A clean coarsening of
frame-sigs -> completeness 1.0 (each sig in one state) even with fewer states; a
misaligned partition -> lower on both.  Also report per-state and per-sig purity.
"""
import argparse, pickle, collections, math
import numpy as np
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary


def sig(s):
    wc = 0; f0 = f1 = False
    for c in s:
        if c >= 3: wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return "SINK" if (f0 and f1) else (f0, f1, wc % 3)


def entropy(counts):
    n = sum(counts)
    return -sum((c / n) * math.log(c / n) for c in counts if c > 0) if n else 0.0


def cond_entropy(joint, cond_totals):
    # H(X|Y) = sum_y p(y) H(X|y)
    n = sum(cond_totals.values())
    h = 0.0
    for y, sub in joint.items():
        py = cond_totals[y] / n
        h += py * entropy(list(sub.values()))
    return h


def vmeasure(states, sigs):
    # H(sig|state) and H(state|sig)
    by_state = collections.defaultdict(collections.Counter)
    by_sig = collections.defaultdict(collections.Counter)
    st_tot = collections.Counter(); sg_tot = collections.Counter()
    for s, g in zip(states, sigs):
        by_state[s][g] += 1; by_sig[g][s] += 1
        st_tot[s] += 1; sg_tot[g] += 1
    H_sig = entropy(list(sg_tot.values()))
    H_state = entropy(list(st_tot.values()))
    H_sig_g_state = cond_entropy(by_state, st_tot)
    H_state_g_sig = cond_entropy(by_sig, sg_tot)
    homo = 1 - H_sig_g_state / H_sig if H_sig else 1.0   # states pure in sig
    comp = 1 - H_state_g_sig / H_state if H_state else 1.0  # each sig in one state
    v = 2 * homo * comp / (homo + comp) if (homo + comp) else 0.0
    return homo, comp, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--dump-dir", default="mr_dumps/seed2")
    ap.add_argument("--n-eval", type=int, default=8000)
    args = ap.parse_args()

    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    samp = SuperSampler(vocab, 36)
    rng = np.random.default_rng(args.seed + 999)
    supers = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(args.n_eval)]
    sigs = [sig(w) for w in supers]
    # binary "SINK vs not" reference too
    sink = ["SINK" if g == "SINK" else "live" for g in sigs]

    def walk(dfa, w):
        s = dfa.initial_state
        for c in w: s = dfa.transitions[s][c]
        return s

    r = 0
    while True:
        try:
            rec = pickle.load(open(f"{args.dump_dir}/round_{r:02d}.pkl", "rb"))
        except FileNotFoundError:
            break
        dfa = rec["dfa"]
        states = [walk(dfa, w) for w in supers]
        nst = len(set(states))
        h, c, v = vmeasure(states, sigs)
        hb, cb, vb = vmeasure(states, sink)
        # mean per-state purity (dominant sig share), size-weighted
        by_state = collections.defaultdict(collections.Counter)
        for s, g in zip(states, sigs): by_state[s][g] += 1
        tot = len(states)
        wpur = sum(max(cc.values()) for cc in by_state.values()) / tot
        print(f"seed {args.seed} round {r}: {nst} reached states | "
              f"vs FRAME-SIG: homogeneity {h:.3f} completeness {c:.3f} V {v:.3f} | "
              f"vs SINK/live: homo {hb:.3f} comp {cb:.3f} V {vb:.3f} | mean state purity {wpur:.3f}")
        r += 1


if __name__ == "__main__":
    main()
