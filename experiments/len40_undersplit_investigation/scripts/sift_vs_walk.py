"""Close the gap: the family SIFTS cleanly but the DFA WALK partition is misaligned?

For each round we route a fixed eval set two ways:
  - SIFT: through the discrimination tree with the real oracle band decider (this is
    what the family/tree actually decides -- indecisive strings get no leaf);
  - WALK: follow the exported DFA's transitions symbol by symbol.
We score each partition's homogeneity on SINK-vs-live and on frame-sig, and report the
sift/walk agreement rate (== what est measures).  If SIFT is clean (family works) but
WALK is not, the misalignment lives in transition export, not the family.
"""
import argparse, pickle, collections, math
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.l_star.midfix_tree import oracle_decider
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

NSUF = 32


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


def homogeneity(states, labels):
    by_state = collections.defaultdict(collections.Counter)
    tot = collections.Counter()
    lab_tot = collections.Counter()
    for s, g in zip(states, labels):
        by_state[s][g] += 1; tot[s] += 1; lab_tot[g] += 1
    H_lab = entropy(list(lab_tot.values()))
    n = len(states)
    H_lab_g_state = sum((tot[s] / n) * entropy(list(sub.values())) for s, sub in by_state.items())
    return 1 - H_lab_g_state / H_lab if H_lab else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--dump-dir", default="mr_dumps/seed2")
    ap.add_argument("--n-eval", type=int, default=2500)
    args = ap.parse_args()

    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
    oracle = LiftedOracle(base, vocab, seed=args.seed)

    samp = SuperSampler(vocab, 36)
    rng = np.random.default_rng(args.seed + 999)
    supers = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(args.n_eval)]
    sigs = [sig(w) for w in supers]
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
        dfa, dt = rec["dfa"], rec["dt"]
        # subsample the base family for a cheaper but faithful decider
        fam = dt.base_family
        idx = np.random.default_rng(7).choice(len(fam), min(NSUF, len(fam)), replace=False)
        sub = type(dt).__new__(type(dt))
        sub.__dict__.update(dt.__dict__)
        sub.base_family = [fam[i] for i in idx]
        decide, _ = oracle_decider(oracle, sub.base_family,
                                   accept=rec["accept_thresh"], reject=rec["reject_thresh"])

        sift_state, walk_state, keep_sink, keep_sig = [], [], [], []
        indec = 0
        for w, gk, sg in zip(supers, sink, sigs):
            leaf, _ = sub.sift(bytes(w), decide)
            if leaf is None:
                indec += 1
                continue
            sift_state.append(leaf)
            walk_state.append(walk(dfa, w))
            keep_sink.append(gk); keep_sig.append(sg)

        agree = np.mean([a == b for a, b in zip(sift_state, walk_state)]) if sift_state else float("nan")
        print(f"\nseed {args.seed} round {r}: {rec['n_states']} states, "
              f"{indec}/{args.n_eval} indecisive on sift ({indec/args.n_eval*100:.1f}%), "
              f"sift/walk agree {agree:.3f}  (est was {rec['true_acc']:.3f})")
        print(f"  SINK/live homogeneity:  SIFT {homogeneity(sift_state, keep_sink):.3f}   "
              f"WALK {homogeneity(walk_state, keep_sink):.3f}")
        print(f"  frame-sig  homogeneity:  SIFT {homogeneity(sift_state, keep_sig):.3f}   "
              f"WALK {homogeneity(walk_state, keep_sig):.3f}")
        r += 1


if __name__ == "__main__":
    main()
