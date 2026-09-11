"""End-to-end replication: identical to exp2's pipeline (SuperSampler -> LiftedOracle
-> compile to DNA -> base oracle -> synthesis), with ONLY the base oracle swapped
from spliceai to a simple, understood DNA function = frame rule + weak content
substructure.  The query noise comes from the REAL wildcard-fill compilation
(num_compilations=1), not a hash on the super-string.  If this collapses seed-fragilely
at threshold delta, the mechanism is confirmed through the full data pipeline."""
import argparse, numpy as np
from orthogonal_dfa.l_star.counterexample_synthesis import counterexample_driven_synthesis
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.structures import Oracle, _uniform_random
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

STOPS = {(3, 0, 2), (3, 0, 0), (3, 2, 0)}
def frame_state_dna(dna):
    """On the actual compiled DNA: (f0,f1, first_sign) over all reading frames."""
    f0 = f1 = False; first = 0
    for i in range(len(dna) - 2):
        if (dna[i], dna[i+1], dna[i+2]) in STOPS:
            fr = i % 3
            if fr == 0:
                if not f0 and not f1 and first == 0: first = +1
                f0 = True
            elif fr == 1:
                if not f0 and not f1 and first == 0: first = -1
                f1 = True
    return f0, f1, first

class DNASubstructOracle(Oracle):
    """Base oracle over DNA: frame-0-and-1 rule + weak content substructure
    (which frame closed first).  Deterministic per DNA string -> the noise the
    learner sees comes from the wildcard-fill compilation, exactly like spliceai."""
    def __init__(self, delta, base_acc=0.60, base_rej=0.42):
        self.delta, self.ba, self.br = delta, base_acc, base_rej
    @property
    def alphabet_size(self): return 4
    def membership_query(self, dna):
        f0, f1, first = frame_state_dna(dna)
        p = self.br if (f0 and f1) else self.ba
        p = min(max(p + self.delta * first, 0.02), 0.98)
        return _uniform_random(bytes(dna), 0) < p

def kmer_frame_reject(s):
    wc = 0; f0 = f1 = False
    for c in s:
        if c >= 3: wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return f0 and f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta", type=float, default=0.04)
    args = ap.parse_args()
    vocab = KmerVocabulary(kmers=((3,0,2),(3,0,0),(3,2,0)), base_alphabet_size=4)
    base = DNASubstructOracle(args.delta)
    def oracle_creator(_nm, s):
        return LiftedOracle(base, vocab, seed=s)          # REAL compilation pipeline
    pst = build_pst(oracle_creator, min_signal_strength=0.05, seed=args.seed,
                    sampler=SuperSampler(vocab, 36))
    pst.config.fnr_limit = 0.10
    samp = SuperSampler(vocab, 36); rng = np.random.default_rng(args.seed + 999)
    ev = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(4000)]
    tgt = np.array([0.0 if kmer_frame_reject(w) else 1.0 for w in ev])
    def phi(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return 0.0 if a.std()==0 or b.std()==0 else float(np.corrcoef(a, b)[0,1])
    for i, (dfa, dt, ta, bd, _c) in enumerate(
            counterexample_driven_synthesis(pst, acc_threshold=0.98)):
        call = np.array([bool(dfa.accepts_input(w)) for w in ev], float)
        print(f"[E2E delta={args.delta} seed={args.seed} round {i}] {len(dfa.states)} states, "
              f"accept {call.mean():.3f}, phi(DFA,frame-rule) {phi(call,tgt):+.3f}, "
              f"agree {np.mean(call==tgt):.3f}", flush=True)
        break

if __name__ == "__main__":
    main()
