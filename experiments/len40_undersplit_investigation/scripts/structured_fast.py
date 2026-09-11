"""Fast, higher-signal version of the structured-oracle collapse sandbox.

Speed levers vs the original:
  * shorter strings         (--num-symbols, default 16 vs 36)
  * higher signal strength  (--signal: oracle accept/reject probs = 0.5 +/- signal;
                             default 0.30 vs the old 0.10) -> family converges with
                             far fewer suffixes
  * bigger min-signal        (--min-signal-strength 0.2 vs 0.05) -> smaller family
  * lower acc-threshold      (--acc-threshold 0.88 -> ce_pass patience ~24 vs 149)

The oracle is still frame-0-and-1 rule + a weak content substructure (which frame
closed first) of strength `delta`.  Reports boundary/accept_thresh, round-0 states,
phi vs the frame rule, and accept-rate.
"""
import argparse, numpy as np
from orthogonal_dfa.l_star.counterexample_synthesis import counterexample_driven_synthesis
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.structures import Oracle, _uniform_random
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

def frame_state(s):
    wc = 0; f0 = f1 = False; first = 0
    for c in s:
        if c >= 3: wc += 1
        else:
            ph = wc % 3
            if ph == 0:
                if not f0 and not f1 and first == 0: first = +1
                f0 = True
            elif ph == 1:
                if not f0 and not f1 and first == 0: first = -1
                f1 = True
    return f0, f1, first

class StructuredOracle(Oracle):
    def __init__(self, signal, delta, seed):
        self.acc = 0.5 + signal; self.rej = 0.5 - signal
        self.delta = delta; self.seed = seed
    @property
    def alphabet_size(self): return 5
    def membership_query(self, s):
        f0, f1, first = frame_state(s)
        p = self.rej if (f0 and f1) else self.acc
        p = min(max(p + self.delta * first, 0.02), 0.98)
        return _uniform_random(bytes(s), self.seed) < p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--signal", type=float, default=0.30)
    ap.add_argument("--num-symbols", type=int, default=16)
    ap.add_argument("--acc-threshold", type=float, default=0.88)
    ap.add_argument("--min-signal-strength", type=float, default=0.20)
    ap.add_argument("--fnr-limit", type=float, default=0.10)
    ap.add_argument("--split-pval", type=float, default=0.001)
    ap.add_argument("--n-eval", type=int, default=2000)
    args = ap.parse_args()
    vocab = KmerVocabulary(kmers=((3,0,2),(3,0,0),(3,2,0)), base_alphabet_size=4)
    pst = build_pst(lambda _nm, s: StructuredOracle(args.signal, args.delta, s),
                    min_signal_strength=args.min_signal_strength, seed=args.seed,
                    sampler=SuperSampler(vocab, args.num_symbols))
    pst.config.fnr_limit = args.fnr_limit
    pst.config.split_pval = args.split_pval
    samp = SuperSampler(vocab, args.num_symbols); rng = np.random.default_rng(args.seed + 999)
    ev = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(args.n_eval)]
    tgt = np.array([0.0 if (frame_state(w)[0] and frame_state(w)[1]) else 1.0 for w in ev])
    def phi(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return 0.0 if a.std()==0 or b.std()==0 else float(np.corrcoef(a, b)[0,1])
    for i, (dfa, dt, ta, bd, _c) in enumerate(
            counterexample_driven_synthesis(pst, acc_threshold=args.acc_threshold)):
        call = np.array([bool(dfa.accepts_input(w)) for w in ev], float)
        print(f"[sig={args.signal} delta={args.delta} seed={args.seed} L={args.num_symbols} "
              f"sp={args.split_pval}] "
              f"boundary={pst.decision_boundary:.3f} accept_thresh={pst.accept_thresh:.3f} | "
              f"{len(dfa.states)} states, accept {call.mean():.3f}, "
              f"phi(DFA,frame) {phi(call,tgt):+.3f}, agree {np.mean(call==tgt):.3f}", flush=True)
        break

if __name__ == "__main__":
    main()
