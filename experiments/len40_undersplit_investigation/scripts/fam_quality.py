"""Is round 0's suffix family actually BETTER than round 1's -- on more than seed 2?

The paradox "round 0 has the better family yet splits worse" needs the SETUP (round 0
family better) to hold across seeds, not just the outcome (round 1 structure better). Here,
for each seed with both rounds (2, 3), compare round-0 vs round-1 family on the eval set:
FPR (SINK accepted), FNR (live rejected), indecision -- the metrics that made round 0
"seem better". If round 0 is better on BOTH seeds, the paradox replicates; if round 0 is
worse on seed 3, seed 3 is just "better family wins" and seed 2 stands alone.
"""
import pickle
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

BASE = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad"
SEEDS = {2: f"{BASE}/mr_dumps/seed2", 3: f"{BASE}/mr_multi/seed3"}
N = 1500
NSUF = 40


def sig(s):
    wc = 0; f0 = f1 = False
    for c in s:
        if c >= 3: wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return "SINK" if (f0 and f1) else "live"


def main():
    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
    print(f"{'seed':4} {'rnd':3} {'band':17} {'FPR(SINK->A)':12} {'FNR(live->R)':12} "
          f"{'indec_all':9} {'indec_SINK':10} {'indec_live':10} {'famsize':7}")
    for seed, d in SEEDS.items():
        oracle = LiftedOracle(base, vocab, seed=seed)
        samp = SuperSampler(vocab, 36); rng = np.random.default_rng(seed + 999)
        supers = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(N)]
        sk = np.array([sig(w) == "SINK" for w in supers])
        for rd in (0, 1):
            rec = pickle.load(open(f"{d}/round_{rd:02d}.pkl", "rb"))
            fam = [bytes(v) for v in rec["dt"].base_family]
            vs = [fam[i] for i in np.random.default_rng(7).choice(len(fam), min(NSUF, len(fam)), replace=False)]
            at, rt = rec["accept_thresh"], rec["reject_thresh"]
            combos = [bytes(s) + v for s in supers for v in vs]
            m = np.asarray(oracle.membership_queries(combos), float).reshape(len(supers), len(vs)).mean(1)
            acc = m >= at; rej = m < rt; ind = (~acc) & (~rej)
            fpr = np.mean(acc[sk]) * 100         # SINK accepted
            fnr = np.mean(rej[~sk]) * 100        # live rejected
            print(f"{seed:4} {rd:3} [{rt:.3f},{at:.3f})   {fpr:10.1f}   {fnr:10.1f}   "
                  f"{np.mean(ind)*100:7.1f}  {np.mean(ind[sk])*100:8.1f}   {np.mean(ind[~sk])*100:8.1f}    {len(vs):3d}",
                  flush=True)


if __name__ == "__main__":
    main()
