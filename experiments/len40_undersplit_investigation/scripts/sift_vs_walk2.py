"""Sift vs walk, full family, batched (no subsample confound).

Uses _oracle_classify -> classify_many with the WHOLE base family (one oracle call
per tree level) so the sift partition is exactly what construction's tree decides.
Compares SIFT (tree routing) and WALK (DFA transitions) partitions on SINK/live and
frame-sig, plus indecision and sift/walk agreement.
"""
import argparse, pickle, collections, math
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.l_star.lstar import _oracle_classify
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
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


def homogeneity(states, labels):
    by = collections.defaultdict(collections.Counter); tot = collections.Counter(); lt = collections.Counter()
    for s, g in zip(states, labels):
        by[s][g] += 1; tot[s] += 1; lt[g] += 1
    H = entropy(list(lt.values())); n = len(states)
    Hc = sum((tot[s] / n) * entropy(list(sub.values())) for s, sub in by.items())
    return 1 - Hc / H if H else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--dump-dir", default="mr_dumps/seed2")
    ap.add_argument("--n-eval", type=int, default=3000)
    args = ap.parse_args()

    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
    oracle = LiftedOracle(base, vocab, seed=args.seed)

    samp = SuperSampler(vocab, 36)
    rng = np.random.default_rng(args.seed + 999)
    supers = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(args.n_eval)]
    seqs = [bytes(w) for w in supers]
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
        _, classify_many = _oracle_classify(dt, oracle,
                                             accept=rec["accept_thresh"], reject=rec["reject_thresh"])
        leaves = classify_many(seqs)  # full family, batched; None = indecisive
        ss, ws, ks, kg = [], [], [], []
        indec = 0
        for w, leaf, gk, sg in zip(supers, leaves, sink, sigs):
            if leaf is None:
                indec += 1; continue
            ss.append(leaf); ws.append(walk(dfa, w)); ks.append(gk); kg.append(sg)
        agree = np.mean([a == b for a, b in zip(ss, ws)]) if ss else float("nan")
        print(f"\nseed {args.seed} round {r}: {rec['n_states']} states, "
              f"{indec}/{args.n_eval} indecisive on sift ({indec/args.n_eval*100:.1f}%), "
              f"sift/walk agree {agree:.3f} (est {rec['true_acc']:.3f})")
        print(f"  SINK/live homogeneity:  SIFT {homogeneity(ss, ks):.3f}   WALK {homogeneity(ws, ks):.3f}")
        print(f"  frame-sig  homogeneity:  SIFT {homogeneity(ss, kg):.3f}   WALK {homogeneity(ws, kg):.3f}")
        r += 1


if __name__ == "__main__":
    main()
