"""Test: is round 1's DFA cleaner BECAUSE its family is less decisive?

Mechanism under test: the DT splits a target sig into multiple states only when the
family gives INCONSISTENT DECISIVE labels within that sig (some strings decisive-A,
some decisive-R -> a real distinction the DT must honour).  If instead the family
abstains (indecisive band), the sig stays merged; a consistent-but-wrong label just
gets fixed by denoise.  So we measure, per target sig, under each round's family+band:
  %decisive-accept, %decisive-reject, %indecisive, and the STRADDLE = min(%A,%R)
  among the decisive strings (0 = one-sided, ~0.5 = fully split).
Prediction if the mechanism holds: round 0 shows higher within-sig straddle (fuels
its fragmentation), round 1 shows more indecision / less straddle (stays merged).
"""
import pickle
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

SEED = 2
NPER = 300
NSUF = 60


def sig(s):
    wc = 0; f0 = f1 = False
    for c in s:
        if c >= 3: wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return "SINK" if (f0 and f1) else (f0, f1, wc % 3)


def sstr(g):
    return "SINK" if g == "SINK" else f"f0={'T' if g[0] else 'F'},f1={'T' if g[1] else 'F'},ph{g[2]}"


def main():
    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
    lifted = LiftedOracle(base, vocab, seed=SEED)

    samp = SuperSampler(vocab, 36); rng = np.random.default_rng(12345)
    keys = [(a, b, p) for a in (False, True) for b in (False, True) for p in range(3) if not (a and b)] + ["SINK"]
    pools = {}
    drawn = 0
    while drawn < 300000 and any(len(pools.get(k, [])) < NPER for k in keys):
        w = list(samp.sample(rng, vocab.alphabet_size)); drawn += 1
        g = sig(w); pools.setdefault(g, [])
        if len(pools[g]) < NPER:
            pools[g].append(w)
    order = sorted(pools, key=lambda k: (k == "SINK", str(k)))

    recs = []
    r = 0
    while True:
        try:
            recs.append(pickle.load(open(f"/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/mr_dumps/seed{SEED}/round_{r:02d}.pkl", "rb"))); r += 1
        except FileNotFoundError:
            break

    for ri, rec in enumerate(recs):
        fam = [list(map(int, v)) for v in rec["dt"].base_family]
        vsuf = [fam[i] for i in np.random.default_rng(7).choice(len(fam), min(NSUF, len(fam)), replace=False)]
        at, rt = rec["accept_thresh"], rec["reject_thresh"]
        print(f"\n=== round {ri}: {rec['n_states']} st, band [{rt:.3f},{at:.3f}), boundary {rec['boundary']:.3f} ===")
        print("  target-sig             %A   %R   %indec   straddle=min(%A,%R)/dec")
        for g in order:
            strs = pools[g]
            combos = [bytes(s + v) for s in strs for v in vsuf]
            lab = np.asarray(lifted.membership_queries(combos), float).reshape(len(strs), len(vsuf))
            m = lab.mean(1)
            pa = np.mean(m >= at); pr = np.mean(m < rt); pi = np.mean((m >= rt) & (m < at))
            dec = pa + pr
            straddle = (min(pa, pr) / dec) if dec > 0 else 0.0
            print(f"  {sstr(g):20} {pa*100:4.0f} {pr*100:4.0f}  {pi*100:5.0f}     {straddle:.2f}", flush=True)


if __name__ == "__main__":
    main()
