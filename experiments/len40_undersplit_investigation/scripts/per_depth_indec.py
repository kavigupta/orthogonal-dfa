"""Confirm the depth-accumulation mechanism: does round 0's higher cumulative indecision
(47% vs round 1's 27%) come from its DEEPER sift tree?

Sift the eval set through each tree tracking the depth at which each string drops out as
indecisive (hits the band at that node). Report, per depth d:
  - reached:  strings that reach a node at depth d
  - dropped:  of those, how many go indecisive AT depth d
  - per-node abstain rate = dropped/reached
  - cumulative indecision through depth d
plus the max depth. If round 0 accumulates more indecision purely by having MORE depth
(longer paths) at a similar per-node rate, that confirms deeper-tree -> more indecision.
Full family (matches leaf_purity's 47%/27%).
"""
import pickle, collections
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

SEED = 2; N = 1500
D = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/mr_dumps/seed2"


def main():
    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
    oracle = LiftedOracle(base, vocab, seed=SEED)
    samp = SuperSampler(vocab, 36); rng = np.random.default_rng(SEED + 999)
    supers = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(N)]

    for rd in (0, 1):
        rec = pickle.load(open(f"{D}/round_{rd:02d}.pkl", "rb"))
        dt = rec["dt"]; fam = [bytes(v) for v in dt.base_family]
        at, rt = rec["accept_thresh"], rec["reject_thresh"]
        reached = collections.Counter(); dropped = collections.Counter(); placed = [0]

        def rec_sift(node, idxs, depth):
            if not isinstance(node, tuple):
                placed[0] += len(idxs); return
            midfix, lk = node
            reached[depth] += len(idxs)
            combos = [bytes(supers[i]) + midfix + v for i in idxs for v in fam]
            m = np.asarray(oracle.membership_queries(combos), float).reshape(len(idxs), len(fam)).mean(1)
            t_idx = [idxs[k] for k in range(len(idxs)) if m[k] >= at]
            f_idx = [idxs[k] for k in range(len(idxs)) if m[k] < rt]
            dropped[depth] += len(idxs) - len(t_idx) - len(f_idx)
            rec_sift(lk[True], t_idx, depth + 1)
            rec_sift(lk[False], f_idx, depth + 1)

        rec_sift(dt._root, list(range(N)), 0)
        total_drop = sum(dropped.values())
        maxd = max(reached) if reached else 0
        print(f"\n=== round {rd}: {rec['n_states']} states, max sift depth {maxd + 1}, "
              f"total indecision {total_drop}/{N} = {total_drop/N*100:.1f}% ===", flush=True)
        print("  depth  reached  dropped  per-node%  cum-indec%")
        cum = 0
        for d in range(maxd + 1):
            cum += dropped[d]
            rr = reached[d]; dd = dropped[d]
            print(f"   {d:3d}   {rr:6d}   {dd:6d}   {(dd/rr*100 if rr else 0):6.1f}     {cum/N*100:6.1f}", flush=True)


if __name__ == "__main__":
    main()
