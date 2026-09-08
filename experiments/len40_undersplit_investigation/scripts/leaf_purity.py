"""Leaf purity at the SINK/live boundary -- the thing that feeds edge resolution.

EdgeResolver picks the first decisively-sifting MEMBER of a leaf and points the edge there;
an impure leaf (mixed SINK/live members) -> heterogeneous members -> arbitrary edge. So the
quantity that matters is how pure the tree LEAVES are in SINK/live, and specifically how
much SINK contaminates the ACCEPT-side leaves (the contamination that becomes the bad edge
trapping SINK in the accept-cycle).

For round 0 and round 1 (seed 2): sift the eval set through each tree, report
  - indecision rate
  - among decisive: overall SINK/live homogeneity (leaf purity)
  - accept-side leaves: total members, and SINK contamination (SINK on accept side)
  - reject-side leaves: live contamination
so round-0 vs round-1 leaf purity is compared on equal footing.
"""
import pickle, collections, math
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.l_star.lstar import _oracle_classify
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

SEED = 2
NEVAL = 3000
D = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/mr_dumps/seed2"


def sig(s):
    wc = 0; f0 = f1 = False
    for c in s:
        if c >= 3: wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return "SINK" if (f0 and f1) else "live"


def entropy(cs):
    n = sum(cs)
    return -sum((c / n) * math.log(c / n) for c in cs if c > 0) if n else 0.0


def homogeneity(states, labels):
    by = collections.defaultdict(collections.Counter); tot = collections.Counter(); lt = collections.Counter()
    for s, g in zip(states, labels):
        by[s][g] += 1; tot[s] += 1; lt[g] += 1
    H = entropy(list(lt.values())); n = len(states)
    Hc = sum((tot[s] / n) * entropy(list(sub.values())) for s, sub in by.items())
    return 1 - Hc / H if H else 1.0


def main():
    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
    oracle = LiftedOracle(base, vocab, seed=SEED)
    samp = SuperSampler(vocab, 36); rng = np.random.default_rng(SEED + 999)
    supers = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(NEVAL)]
    seqs = [bytes(w) for w in supers]
    sk = [sig(w) for w in supers]

    for rd in (0, 1):
        rec = pickle.load(open(f"{D}/round_{rd:02d}.pkl", "rb"))
        dt, dfa = rec["dt"], rec["dfa"]
        acc = getattr(dfa, "final_states", set())
        _, classify_many = _oracle_classify(dt, oracle,
                                             accept=rec["accept_thresh"], reject=rec["reject_thresh"])
        leaves = classify_many(seqs)
        dec = [(l, g) for l, g in zip(leaves, sk) if l is not None]
        indec = 1 - len(dec) / len(leaves)
        ls = [l for l, g in dec]; gs = [g for l, g in dec]
        homog = homogeneity(ls, gs)
        # accept-side vs reject-side leaf contamination
        acc_S = sum(1 for l, g in dec if l in acc and g == "SINK")
        acc_L = sum(1 for l, g in dec if l in acc and g == "live")
        rej_S = sum(1 for l, g in dec if l not in acc and g == "SINK")
        rej_L = sum(1 for l, g in dec if l not in acc and g == "live")
        accN = acc_S + acc_L; rejN = rej_S + rej_L
        print(f"\n=== round {rd}: {rec['n_states']} states, band [{rec['reject_thresh']:.3f},{rec['accept_thresh']:.3f}) ===")
        print(f"  indecision {indec*100:.1f}%  |  decisive SINK/live homogeneity {homog:.3f}")
        print(f"  ACCEPT-side leaves: {accN} members, SINK contamination {acc_S} ({acc_S/max(accN,1)*100:.1f}%)")
        print(f"  REJECT-side leaves: {rejN} members, live contamination {rej_L} ({rej_L/max(rejN,1)*100:.1f}%)")
        # per accept-side leaf, its SINK fraction (the impurity feeding its edges)
        byl = collections.defaultdict(lambda: {"SINK": 0, "live": 0})
        for l, g in dec:
            byl[l][g] += 1
        imp = [(l, byl[l]) for l in byl if l in acc and min(byl[l]["SINK"], byl[l]["live"]) >= 5]
        imp.sort(key=lambda x: -min(x[1]["SINK"], x[1]["live"]))
        print(f"  impure ACCEPT leaves (both>=5): " +
              ", ".join("s%d(%dS/%dL)" % (l, c["SINK"], c["live"]) for l, c in imp[:8]))


if __name__ == "__main__":
    main()
