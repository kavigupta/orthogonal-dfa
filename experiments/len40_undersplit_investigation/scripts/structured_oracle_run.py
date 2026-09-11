"""Constructive test of the theory: a synthetic oracle = frame-0-and-1 rule (weak
dominant signal) PLUS a tunable weak content-correlated substructure that breaks
the frame-symmetric equivalence (delta).  The substructure feature is "which frame
closed first" -- a persistent property of the prefix content, exactly what
distinguishes the (T,F,.)/(F,T,.) pairs, mimicking how spliceai tells them apart.

Theory predicts: delta=0 -> clean (no over-split); delta>0 -> over-split + accept-all
collapse, seed-fragile -- replicating spliceai."""
import argparse, numpy as np
from orthogonal_dfa.l_star.counterexample_synthesis import counterexample_driven_synthesis
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.structures import Oracle, _uniform_random  # per-string hash
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

def frame_state(s):
    """(f0_closed, f1_closed, first_sign): first_sign = +1 if a phase-0 kmer precedes
    any phase-1 kmer, -1 if the reverse, 0 if neither frame closed."""
    wc = 0; f0 = f1 = False; first = 0
    for c in s:
        if c >= 3:
            wc += 1
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
    def __init__(self, delta, seed, base_acc=0.60, base_rej=0.42):
        self.delta, self.seed, self.ba, self.br = delta, seed, base_acc, base_rej
    @property
    def alphabet_size(self): return 5
    def membership_query(self, string):
        f0, f1, first = frame_state(string)
        reject = f0 and f1
        p = self.br if reject else self.ba
        p += self.delta * first          # weak substructure breaking (T,F,.)/(F,T,.)
        p = min(max(p, 0.02), 0.98)
        return _uniform_random(bytes(string), self.seed) < p   # deterministic per string

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta", type=float, default=0.0)
    args = ap.parse_args()
    vocab = KmerVocabulary(kmers=((3,0,2),(3,0,0),(3,2,0)), base_alphabet_size=4)
    def oracle_creator(_nm, s): return StructuredOracle(args.delta, s)
    pst = build_pst(oracle_creator, min_signal_strength=0.05, seed=args.seed,
                    sampler=SuperSampler(vocab, 36))
    pst.config.fnr_limit = 0.10
    samp = SuperSampler(vocab, 36); rng = np.random.default_rng(args.seed + 999)
    ev = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(4000)]
    tgt = np.array([0.0 if (frame_state(w)[0] and frame_state(w)[1]) else 1.0 for w in ev])
    def phi(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return 0.0 if a.std()==0 or b.std()==0 else float(np.corrcoef(a, b)[0,1])
    for i, (dfa, dt, ta, bd, _c) in enumerate(
            counterexample_driven_synthesis(pst, acc_threshold=0.98)):
        call = np.array([bool(dfa.accepts_input(w)) for w in ev], float)
        print(f"[delta={args.delta} seed={args.seed} round {i}] {len(dfa.states)} states, "
              f"accept {call.mean():.3f}, phi(DFA,frame-rule) {phi(call,tgt):+.3f}, "
              f"agree {np.mean(call==tgt):.3f}", flush=True)
        break

if __name__ == "__main__":
    main()
