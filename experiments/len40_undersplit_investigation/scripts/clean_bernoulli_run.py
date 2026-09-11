"""Replication test: run the SAME synthesis over a CLEAN oracle whose label is
exactly the frame-0-and-1 rule (deterministic from the super-string's kmers) plus
per-string symmetric-Bernoulli hash noise.  If this does NOT over-split/collapse,
the spliceai collapse is due to spliceai's real sub-structure, not the split test."""
import argparse, numpy as np
from orthogonal_dfa.l_star.counterexample_synthesis import counterexample_driven_synthesis
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.structures import Oracle, SymmetricBernoulli
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

def frame01_reject_from_kmers(s):
    """Deterministic true label from the super-string's kmers: reject iff a kmer
    lands in frame 0 AND a kmer lands in frame 1 (phase = #wildcards-so-far)."""
    wc = 0; f0 = f1 = False
    for c in s:
        if c >= 3:
            wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return f0 and f1

class CleanFrameOracle(Oracle):
    def __init__(self, p_correct, seed):
        self._noise = SymmetricBernoulli(p_correct=p_correct); self._seed = seed
    @property
    def alphabet_size(self): return 5
    def membership_query(self, string):
        true_accept = not frame01_reject_from_kmers(string)
        return self._noise.apply_noise(true_accept, bytes(string), self._seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-correct", type=float, default=0.65)
    ap.add_argument("--fnr-limit", type=float, default=0.10)
    args = ap.parse_args()
    vocab = KmerVocabulary(kmers=((3,0,2),(3,0,0),(3,2,0)), base_alphabet_size=4)
    def oracle_creator(_nm, s):
        return CleanFrameOracle(args.p_correct, s)
    pst = build_pst(oracle_creator, min_signal_strength=0.05, seed=args.seed,
                    sampler=SuperSampler(vocab, 36))
    pst.config.fnr_limit = args.fnr_limit
    print(f"[clean p_correct={args.p_correct} seed={args.seed}] synthesizing...", flush=True)

    # eval vs the exact frame rule
    samp = SuperSampler(vocab, 36); rng = np.random.default_rng(args.seed + 999)
    ev = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(4000)]
    tgt = np.array([0.0 if frame01_reject_from_kmers(w) else 1.0 for w in ev])  # 1=accept
    def phi(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return 0.0 if a.std()==0 or b.std()==0 else float(np.corrcoef(a, b)[0,1])
    for i, (dfa, dt, true_acc, boundary, _c) in enumerate(
            counterexample_driven_synthesis(pst, acc_threshold=0.98)):
        call = np.array([bool(dfa.accepts_input(w)) for w in ev], float)
        print(f"[clean p={args.p_correct} seed={args.seed} round {i}] "
              f"{len(dfa.states)} states, accept {call.mean():.3f}, "
              f"phi(DFA,frame-rule) {phi(call,tgt):+.3f}, agree {np.mean(call==tgt):.3f}", flush=True)
        break   # round 0 only

if __name__ == "__main__":
    main()
