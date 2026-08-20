"""Analyze the round DFAs dumped by exp2_learn.py --dump-dir.

Scores with phi (correlation) and mutual information, NOT est (est rewards
accept-all).  For each round_NN.pkl:
  * phi(DFA, oracle) and phi(DFA, all-frames-closed);
For the best round's DFA it also:
  * prints the transition table (symbols named TAG/TAA/TGA/X/Y);
  * finds exactly which sets of closed reading frames it rejects; and
  * compares every frame predicate to the oracle by phi and MI, so you can see
    whether the DFA's predicate beats "all frames closed".

Result (seed-0 dumps): round 0 = 8 states, phi(DFA,oracle)=+0.167; the DFA rejects
exactly when frames 0 AND 1 are both closed.  That predicate scores phi=-0.158 /
MI=0.019 bits with the oracle, vs -0.088 / 0.0048 for all-frames-closed -- ~1.8x
(phi) and ~4x (MI) stronger, and the best single binary frame predicate.
"""
import argparse
import glob
import os
import pickle
import numpy as np

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage import KmerVocabulary, SuperSampler

TAG, TAA, TGA = (3, 0, 2), (3, 0, 0), (3, 2, 0)
STOPS = {TAG, TAA, TGA}
NAME = {0: "TAG", 1: "TAA", 2: "TGA", 3: "X", 4: "Y"}


def frame_closed(seq, ph):
    sub = seq[ph:]
    return any(tuple(sub[i:i+3]) in STOPS for i in range(0, len(sub)-2, 3))


def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def mi_bits(x, y):
    x, y = np.asarray(x), np.asarray(y)
    xs = {v: i for i, v in enumerate(np.unique(x))}
    ys = {v: i for i, v in enumerate(np.unique(y))}
    j = np.zeros((len(xs), len(ys)))
    for a, b in zip(x, y):
        j[xs[a], ys[b]] += 1
    j /= j.sum()
    px, py = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = j * (np.log2(j) - np.log2(px) - np.log2(py))
    return float(np.nansum(t))


def print_dfa(dfa):
    print(f"  states {sorted(dfa.states)}, initial {dfa.initial_state}, "
          f"accepting {sorted(dfa.final_states)}")
    for s in sorted(dfa.states):
        row = dfa.transitions[s]
        parts = [f"{NAME.get(int(k), k)}->{row[k]}" for k in sorted(row, key=lambda k: int(k))]
        mark = "ACCEPT" if s in dfa.final_states else "reject"
        init = " (init)" if s == dfa.initial_state else ""
        print(f"    {s}{init} [{mark}]: " + "  ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--len-lo", type=int, default=35)
    ap.add_argument("--len-hi", type=int, default=85)
    ap.add_argument("--num-symbols", type=int, default=36)
    ap.add_argument("--n-eval", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    base = gate_residual_oracle(
        default_exon, load_spliceai(400, 0),
        length=args.length, len_lo=args.len_lo, len_hi=args.len_hi,
    )
    vocab = KmerVocabulary(kmers=(TAG, TAA, TGA), base_alphabet_size=4)
    samp = SuperSampler(vocab, args.num_symbols)
    rng = np.random.default_rng(args.seed)
    ws = [samp.sample(rng, vocab.alphabet_size) for _ in range(args.n_eval)]
    bs = vocab.compile_many(ws, [np.random.default_rng(i) for i in range(args.n_eval)])
    ora = np.asarray(base.membership_queries(bs)).astype(int)
    fc = np.array([[frame_closed(b, p) for p in range(3)] for b in bs], dtype=int)
    afc = fc.all(1).astype(int)
    print(f"eval N={args.n_eval}, oracle accept {ora.mean():.3f}, "
          f"phi(oracle, all-frames-closed) = {phi(ora, afc):+.3f}  [baseline ceiling]\n")

    best = None
    for path in sorted(glob.glob(os.path.join(args.dump_dir, "round_*.pkl"))):
        d = pickle.load(open(path, "rb"))
        dfa = d.get("dfa")
        if dfa is None:
            print(f"{os.path.basename(path)}: no dfa"); continue
        call = np.array([bool(dfa.accepts_input(w)) for w in ws], dtype=int)
        pdo = phi(call, ora)
        print(f"{os.path.basename(path)}: round {d.get('round')}, {len(dfa.states)} states, "
              f"est {d.get('est'):.3f}, accept {call.mean():.3f}  |  "
              f"phi(DFA,oracle)={pdo:+.3f}  phi(DFA,afc)={phi(call, afc):+.3f}")
        if best is None or abs(pdo) > abs(best[1]):
            best = (path, pdo, dfa, call)

    if best is None:
        return
    path, _, dfa, call = best
    print(f"\n=== best round: {os.path.basename(path)} ===")
    print_dfa(dfa)

    # which sets of closed frames does it reject?
    key = fc[:, 0] * 4 + fc[:, 1] * 2 + fc[:, 2]
    print("\n  reject-rate by exact set of closed frames:")
    for k in range(8):
        m = key == k
        if m.sum():
            fs = "{" + ",".join(str(i) for i in range(3) if k & (4 >> i)) + "}"
            print(f"    {fs:<9}: reject {1-call[m].mean():.3f}  (n={int(m.sum())})")

    # frame predicates vs oracle: phi and MI
    preds = {
        "frame0 closed": fc[:, 0], "frame1 closed": fc[:, 1], "frame2 closed": fc[:, 2],
        "frames {0,1} both": fc[:, 0] & fc[:, 1], "frames {0,2} both": fc[:, 0] & fc[:, 2],
        "frames {1,2} both": fc[:, 1] & fc[:, 2], "ALL frames closed": afc,
        "count frames closed": fc.sum(1), "full (f0,f1,f2)": key, "DFA-accept": call,
    }
    print("\n  predicate vs oracle:            phi        MI(bits)")
    for name, p in preds.items():
        print(f"    {name:22s} {phi(ora, p):+.3f}     {mi_bits(ora, p):.5f}")


if __name__ == "__main__":
    main()
