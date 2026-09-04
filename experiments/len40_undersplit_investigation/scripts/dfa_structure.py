"""Describe the concrete DFA learned at each round.

For each round's saved DFA we:
  - denoise it (so we see the accept labels the pipeline would actually ship);
  - walk a fixed eval set, tagging every string with its target frame signature
    sig(s) = (f0-closed, f1-closed, phase), SINK if both frames closed;
  - per PRODUCED state: n strings reaching it, its dominant target sig + purity,
    the oracle accept-rate of strings there, and raw/denoised accept labels;
  - per TARGET sig: how it is spread across produced states (fragmentation);
  - the transition table, each state annotated by its dominant sig.
This says whether the round learned the frame automaton, over-split it, or
collapsed it.
"""
import argparse, pickle, types, collections
import numpy as np
import scipy.stats  # noqa: F401
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.l_star.lstar import denoise_accept_labels
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

STOPS = {(3, 0, 2), (3, 0, 0), (3, 2, 0)}


def sig(s):
    wc = 0; f0 = f1 = False
    for c in s:
        if c >= 3:
            wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return "SINK" if (f0 and f1) else (f0, f1, wc % 3)


def sstr(g):
    if g == "SINK":
        return "SINK"
    f0, f1, ph = g
    return f"f0={'T' if f0 else 'F'},f1={'T' if f1 else 'F'},ph{ph}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--dump-dir", default="mr_dumps/seed2")
    ap.add_argument("--n-eval", type=int, default=6000)
    args = ap.parse_args()

    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
    oracle = LiftedOracle(base, vocab, seed=args.seed)

    samp = SuperSampler(vocab, 36)
    rng = np.random.default_rng(args.seed + 999)
    supers = [samp.sample(rng, vocab.alphabet_size) for _ in range(args.n_eval)]
    bases = vocab.compile_many(supers, [np.random.default_rng(i) for i in range(args.n_eval)])
    ora = np.asarray(base.membership_queries([bytes(b) for b in bases])).astype(float)
    # target sig is the SUPERLANGUAGE state of the super-string (4/5 = wildcards that
    # shift the reading frame); applying sig() to a compiled base misreads nt 3 as a
    # wildcard, so it must run on the super-string, not the base.
    sigs = [sig(w) for w in supers]
    A = vocab.alphabet_size

    def walk(dfa, w):
        s = dfa.initial_state
        for c in w:
            s = dfa.transitions[s][c]
        return s

    r = 0
    while True:
        try:
            rec = pickle.load(open(f"{args.dump_dir}/round_{r:02d}.pkl", "rb"))
        except FileNotFoundError:
            break
        dfa = rec["dfa"]
        raw_final = set(dfa.final_states)

        cfg = types.SimpleNamespace(min_signal_strength=0.05)
        tbl = types.SimpleNamespace(prefixes=[list(p) for p in rec["prefixes"]])
        pst = types.SimpleNamespace(
            sampler=samp, config=cfg, decision_boundary=rec["boundary"],
            alphabet_size=A, rng=np.random.default_rng(1234 + r), table=tbl, oracle=oracle,
        )
        dn = denoise_accept_labels(pst, dfa)
        dn_final = set(dn.final_states)

        st = np.array([walk(dfa, w) for w in supers])
        reached = sorted(set(st.tolist()))
        by_state_sig = collections.defaultdict(collections.Counter)
        for s, g in zip(st, sigs):
            by_state_sig[s][g] += 1

        print(f"\n{'='*78}\nseed {args.seed} ROUND {r}: {len(dfa.states)} states "
              f"({len(reached)} reached), boundary {rec['boundary']:.3f}, est {rec['true_acc']:.3f}")
        print(f"{'='*78}")
        print(" state  raw dn   n     oracle%  dominant-sig (purity)   other sigs")
        for s in reached:
            cc = by_state_sig[s]; n = sum(cc.values())
            dom, dn_ct = cc.most_common(1)[0]
            purity = dn_ct / n
            mask = st == s
            orate = ora[mask].mean()
            raw = "A" if s in raw_final else "R"
            dnl = "A" if s in dn_final else "R"
            flip = " *" if (s in raw_final) != (s in dn_final) else "  "
            others = ", ".join(f"{sstr(g)}:{c}" for g, c in cc.most_common()[1:4])
            print(f"  s{s:<3}  {raw}  {dnl}{flip} {n:5d}  {orate:5.2f}   {sstr(dom):22} {purity*100:3.0f}%  {others}")

        # per-target-sig fragmentation
        print("\n  target-sig -> produced states (fragmentation):")
        by_sig_state = collections.defaultdict(collections.Counter)
        for s, g in zip(st, sigs):
            by_sig_state[g][s] += 1
        for g in sorted(by_sig_state, key=lambda k: (k == "SINK", str(k))):
            cc = by_sig_state[g]; n = sum(cc.values())
            spread = ", ".join(f"s{s}[{'A' if s in dn_final else 'R'}]:{c}" for s, c in cc.most_common()[:5])
            frag = "  <-- SPLIT" if sum(1 for c in cc.values() if c / n > 0.1) > 1 else ""
            print(f"    {sstr(g):22} n={n:4d}: {spread}{frag}")

        # transition table annotated by dominant sig
        dom_of = {}
        for s in reached:
            dom_of[s] = sstr(by_state_sig[s].most_common(1)[0][0])
        print("\n  transitions (annotated by dominant sig; symbols 0-3 bases, 4-5 wildcard):")
        for s in reached:
            tgts = " ".join(f"{c}->s{dfa.transitions[s][c]}" for c in range(A))
            print(f"    s{s:<3} [{dom_of[s]:14}] {tgts}")
        r += 1


if __name__ == "__main__":
    main()
