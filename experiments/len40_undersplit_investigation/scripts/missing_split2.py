"""Correct version: trace s10's conflated SINK/live subsets through each round's ACTUAL
tree (path node-by-node, honouring the conditional structure), find the node where round 1
separates them (the real split), and check whether round 0's tree even contains that
distinguisher.

For the most-conflated round-0 state, we:
  - trace its SINK-subset and live-subset through round 1's tree (fam1+band1): the first
    node where SINK-majority and live-majority take opposite branches is round 1's split.
  - trace the SAME members through round 0's tree (fam0+band0): do they land in one leaf
    (confirming the undersplit)?  Does round 0's tree contain a node with the separating
    midfix at all?
If the separating midfix is ABSENT from round 0's tree, the undersplit is a distinguisher
round 0 never discovered -- and we then ask why (pool coverage / counterexample pass).
"""
import pickle, collections
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

SEED = 2
NEVAL = 4000
NSK = 80
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


def walk(dfa, w):
    s = dfa.initial_state
    for c in w:
        s = dfa.transitions[s][c]
    return s


def all_midfixes(root):
    out = set()
    def rec(node):
        if not isinstance(node, tuple):
            return
        m, lk = node
        out.add(bytes(m)); rec(lk[True]); rec(lk[False])
    rec(root)
    return out


def trace(root, strings, fam, at, rt, oracle, label):
    """Route each string through the tree, recording (midfix, decision) per node.
    Returns leaves[i] and, per tree node visited, the SINK/live branch counts."""
    n = len(strings)
    leaves = [None] * n
    node_log = []  # (depth, midfix, n_true, n_false, n_indec)

    def rec(node, idxs, depth):
        if not isinstance(node, tuple):
            for i in idxs:
                leaves[i] = node
            return
        midfix, lk = node
        combos = [bytes(strings[i]) + midfix + v for i in idxs for v in fam]
        m = np.asarray(oracle.membership_queries(combos), float).reshape(len(idxs), len(fam)).mean(1)
        t_idx, f_idx, ind = [], [], 0
        for k, i in enumerate(idxs):
            if m[k] >= at:
                t_idx.append(i)
            elif m[k] < rt:
                f_idx.append(i)
            else:
                ind += 1
        node_log.append((depth, bytes(midfix), len(t_idx), len(f_idx), ind))
        rec(lk[True], t_idx, depth + 1)
        rec(lk[False], f_idx, depth + 1)

    rec(root, list(range(n)), 0)
    return leaves, node_log


def main():
    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
    oracle = LiftedOracle(base, vocab, seed=SEED)

    r0 = pickle.load(open(f"{DUMP}/round_00.pkl", "rb"))
    r1 = pickle.load(open(f"{DUMP}/round_01.pkl", "rb"))
    dfa0, dt0 = r0["dfa"], r0["dt"]
    dfa1, dt1 = r1["dfa"], r1["dt"]
    fam0 = [bytes(v) for v in dt0.base_family]
    fam1 = [bytes(v) for v in dt1.base_family]
    at0, rt0 = r0["accept_thresh"], r0["reject_thresh"]
    at1, rt1 = r1["accept_thresh"], r1["reject_thresh"]

    samp = SuperSampler(vocab, 36); rng = np.random.default_rng(SEED + 999)
    supers = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(NEVAL)]
    sk = [sig(w) for w in supers]
    st0 = [walk(dfa0, w) for w in supers]
    by = collections.defaultdict(lambda: {"SINK": [], "live": []})
    for w, g, s in zip(supers, sk, st0):
        by[s][g].append(w)
    S, groups = max(by.items(), key=lambda kv: min(len(kv[1]["SINK"]), len(kv[1]["live"])))
    nS, nL = len(groups["SINK"]), len(groups["live"])
    print(f"target undersplit: round-0 s{S}: {nS} SINK + {nL} live", flush=True)
    S_sink = [groups["SINK"][i] for i in np.random.default_rng(1).choice(nS, min(NSK, nS), replace=False)]
    S_live = [groups["live"][i] for i in np.random.default_rng(2).choice(nL, min(NSK, nL), replace=False)]
    members = S_sink + S_live
    is_sink = np.array([True] * len(S_sink) + [False] * len(S_live))

    for tag, root, fam, at, rt in [("ROUND 1", dt1.root, fam1, at1, rt1),
                                   ("ROUND 0", dt0.root, fam0, at0, rt0)]:
        leaves, node_log = trace(root, members, fam, at, rt, oracle, tag)
        leaves = np.array([-1 if l is None else l for l in leaves])
        print(f"\n=== {tag} tree: where s{S}'s members sift ===", flush=True)
        for lf in sorted(set(leaves)):
            m = leaves == lf
            print(f"    leaf {lf:3d}: {int((m & is_sink).sum()):3d} SINK  {int((m & ~is_sink).sum()):3d} live")
        print(f"  path (depth: midfix-len -> #true #false #indec):")
        for depth, mid, nt, nf, ni in node_log:
            print(f"    d{depth} m(len{len(mid)}): T={nt} F={nf} I={ni}")


if __name__ == "__main__":
    main()
