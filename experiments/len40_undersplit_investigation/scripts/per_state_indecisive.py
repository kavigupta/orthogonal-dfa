"""Controlled per-target-state family diagnostic.

For each target state sig(s)=(f0-kmer-closed,f1-kmer-closed,phase) (SINK if both),
sample ~1000 strings in that state, and under each saved round's family measure the
fraction that are INDECISIVE -- family-mean over the round's suffixes landing in the
[reject_thresh, accept_thresh) band.  family-mean(s) = fraction of family suffixes v
for which the (lifted) oracle accepts s+v.  Shows, per state, whether the family can
decisively place it and whether that improves round over round.
"""
import argparse, pickle
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

SEED = 2
NPER = 1000       # strings per target state
NSUF = 80         # family suffixes sampled for the mean

def sig(s):
    wc = 0; f0 = f1 = False
    for c in s:
        if c >= 3: wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return "SINK" if (f0 and f1) else (f0, f1, wc % 3)

def main():
    vocab = KmerVocabulary(kmers=((3,0,2),(3,0,0),(3,2,0)), base_alphabet_size=4)
    base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
    lifted = LiftedOracle(base, vocab, seed=SEED)

    # build per-signature string pools
    samp = SuperSampler(vocab, 36); rng = np.random.default_rng(12345)
    pools = {}
    drawn = 0
    while drawn < 400000 and any(len(pools.get(k, [])) < NPER for k in
            [(a,b,p) for a in (False,True) for b in (False,True) for p in range(3) if not (a and b)] + ["SINK"]):
        w = list(samp.sample(rng, vocab.alphabet_size)); drawn += 1
        g = sig(w)
        pools.setdefault(g, [])
        if len(pools[g]) < NPER:
            pools[g].append(w)
    print(f"pools built from {drawn} draws; sizes: " +
          ", ".join(f"{k}:{len(v)}" for k, v in sorted(pools.items(), key=str)), flush=True)

    # rounds
    recs = []
    r = 0
    while True:
        try:
            with open(f"/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/mr_dumps/seed{SEED}/round_{r:02d}.pkl", "rb") as f:
                recs.append(pickle.load(f)); r += 1
        except FileNotFoundError:
            break

    order = sorted(pools, key=lambda k: (k == "SINK", k))
    for ri, rec in enumerate(recs):
        fam = [list(map(int, v)) for v in rec["dt"].base_family]
        vsuf = [fam[i] for i in np.random.default_rng(7).choice(len(fam), min(NSUF, len(fam)), replace=False)]
        at, rt = rec["accept_thresh"], rec["reject_thresh"]
        print(f"\n=== round {ri}: {rec['n_states']} st, phi(oracle) {rec['phi_oracle']:+.3f}, "
              f"boundary {rec['boundary']:.3f}, band [{rt:.3f},{at:.3f}) ===")
        print("  target-state       n     %indec  %acc  %rej   mean-of-means")
        for g in order:
            strs = pools[g]
            # batch: every (string, suffix) pair -> mean per string
            combos = [bytes(s + v) for s in strs for v in vsuf]
            lab = np.asarray(lifted.membership_queries(combos), float).reshape(len(strs), len(vsuf))
            m = lab.mean(1)
            indec = np.mean((m >= rt) & (m < at))
            acc = np.mean(m >= at); rej = np.mean(m < rt)
            print(f"  {str(g):16} {len(strs):4d}   {indec*100:5.1f}  {acc*100:4.0f}  {rej*100:4.0f}   {m.mean():.3f}", flush=True)

if __name__ == "__main__":
    main()
