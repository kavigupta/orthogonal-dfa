"""Does the family go indecisive on SHORTER prefixes?

The per-state diagnostic used full-length (36) strings and found decisive
classification.  But construction sifts prefixes of many lengths.  Here we sweep the
sampled length k and, under each round's family+band, measure the family's decision
breakdown (%decisive-accept / %reject / %indecisive) -- aggregate and split by the
coarse SINK/live target.  If %indecisive climbs as k shrinks, the family cannot place
short prefixes, which is what construction routes on.
"""
import argparse, pickle, collections
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

SEED = 2
NPER = 500
NSUF = 60
LENGTHS = [6, 12, 18, 24, 30, 36]
DUMP = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/mr_dumps/seed2"


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
    lifted = LiftedOracle(base, vocab, seed=SEED)

    recs = []
    r = 0
    while True:
        try:
            recs.append(pickle.load(open(f"{DUMP}/round_{r:02d}.pkl", "rb"))); r += 1
        except FileNotFoundError:
            break

    # sample pools per length once
    pools = {}
    for k in LENGTHS:
        samp = SuperSampler(vocab, k); rng = np.random.default_rng(1000 + k)
        pools[k] = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(NPER)]

    for ri, rec in enumerate(recs):
        fam = [bytes(v) for v in rec["dt"].base_family]
        vsuf = [fam[i] for i in np.random.default_rng(7).choice(len(fam), min(NSUF, len(fam)), replace=False)]
        at, rt = rec["accept_thresh"], rec["reject_thresh"]
        print(f"\n=== round {ri}: band [{rt:.3f},{at:.3f}), boundary {rec['boundary']:.3f} ===", flush=True)
        print("  len |  n  | %A  %R  %indec |  SINK: %A %R %ind (n) | live: %A %R %ind (n)")
        for k in LENGTHS:
            strs = pools[k]
            sgs = [sig(s) for s in strs]
            combos = [bytes(s) + v for s in strs for v in vsuf]
            m = np.asarray(lifted.membership_queries(combos), float).reshape(len(strs), len(vsuf)).mean(1)
            def brk(mask):
                mm = m[mask]
                if len(mm) == 0: return (0, 0, 0, 0)
                return (np.mean(mm >= at) * 100, np.mean(mm < rt) * 100,
                        np.mean((mm >= rt) & (mm < at)) * 100, len(mm))
            a, rj, ind, _ = brk(np.ones(len(m), bool))
            sk = np.array([g == "SINK" for g in sgs]); lv = ~sk
            sa, sr, si, sn = brk(sk); la, lr, li, ln = brk(lv)
            print(f"  {k:3d} | {len(strs):4d}| {a:3.0f} {rj:3.0f}  {ind:4.0f}   |  {sa:3.0f}{sr:4.0f}{si:5.0f} ({sn:4d}) | {la:3.0f}{lr:4.0f}{li:5.0f} ({ln:4d})", flush=True)


if __name__ == "__main__":
    main()
