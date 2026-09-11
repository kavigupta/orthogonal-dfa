"""Evaluate each split IN THE CONTEXT OF THE STATE IT SPLITS (the correct framing).

A split is correct if, among the strings that REACH that node, it separates frame-
INEQUIVALENT members (refines toward frame purity); incorrect if the node is already
frame-PURE and the split subdivides it, or if it cuts within the dominant frame-sig.

Sift the population down each round's tree; at every internal node report:
  - depth, #reaching
  - incoming frame-sig purity (dominant sig fraction) -- is the node already ~pure?
  - whether the split cuts WITHIN the dominant sig (both >=accept and <reject there)
A split of an already-pure node, or one that cuts within its dominant sig, is INCORRECT.
"""
import pickle, collections, os
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

SEED = 2; NPREF = 2000; NSUF = 80
D = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/mr_dumps/seed2"


def finesig(s):
    wc = 0; f0 = f1 = False
    for c in s:
        if c >= 3: wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return "SINK" if (f0 and f1) else (f0, f1, wc % 3)


def fss(t):
    return "SINK" if t == "SINK" else f"({int(t[0])}{int(t[1])}p{t[2]})"


def main():
    ROUND = int(os.environ.get("ROUND", "0"))
    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
    oracle = LiftedOracle(base, vocab, seed=SEED)
    rec = pickle.load(open(f"{D}/round_{ROUND:02d}.pkl", "rb"))
    dt = rec["dt"]; fam = [bytes(v) for v in dt.base_family]
    vs = [fam[i] for i in np.random.default_rng(7).choice(len(fam), min(NSUF, len(fam)), replace=False)]
    at, rt = rec["accept_thresh"], rec["reject_thresh"]

    # varied-length population
    prefs = []
    for L in (6, 12, 18, 24, 30, 36):
        samp = SuperSampler(vocab, L); rng = np.random.default_rng(200 + L)
        prefs += [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(NPREF // 6)]
    sigs = [finesig(p) for p in prefs]

    n_incorrect = [0]; n_total = [0]
    print(f"ROUND {ROUND}: band [{rt:.3f},{at:.3f}), pop {len(prefs)}\n")
    print("  depth  #reach  incoming-purity(dominant)   within-dom-cut?  verdict")

    def rec_sift(node, idxs, depth):
        if not isinstance(node, tuple):
            return
        midfix, lk = node
        n_total[0] += 1
        combos = [bytes(prefs[i]) + midfix + v for i in idxs for v in vs]
        m = np.asarray(oracle.membership_queries(combos), float).reshape(len(idxs), len(vs)).mean(1)
        acc = m >= at; rej = m < rt
        sg = [sigs[i] for i in idxs]
        cnt = collections.Counter(sg)
        dom, dc = cnt.most_common(1)[0]
        purity = dc / len(idxs)
        dom_mask = np.array([x == dom for x in sg])
        da = np.mean(acc[dom_mask]) if dom_mask.any() else 0
        dr = np.mean(rej[dom_mask]) if dom_mask.any() else 0
        within = da >= 0.2 and dr >= 0.2
        # incorrect: node already pure (>=0.8) OR the split cuts within the dominant sig
        incorrect = purity >= 0.8 or within
        n_incorrect[0] += incorrect
        SYM = {0: 'TAG', 1: 'TAA', 2: 'TGA', 3: 'X', 4: 'Y'}
        ren = " ".join(SYM[c] for c in midfix) if midfix else "(empty root)"
        reason = ("subdivides already-pure node" if purity >= 0.8
                  else "cuts within dominant sig" if within else "")
        print(f"   {depth:3d}   {len(idxs):5d}   {fss(dom):8} {purity*100:3.0f}%            "
              f"{'YES' if within else 'no ':4} ({da*100:.0f}A/{dr*100:.0f}R)   "
              f"{'INCORRECT' if incorrect else 'ok':9}  m=[{ren}] {reason}", flush=True)
        t_idx = [idxs[k] for k in range(len(idxs)) if acc[k]]
        f_idx = [idxs[k] for k in range(len(idxs)) if rej[k]]
        rec_sift(lk[True], t_idx, depth + 1)
        rec_sift(lk[False], f_idx, depth + 1)

    rec_sift(dt._root, list(range(len(prefs))), 0)
    print(f"\n  => {n_incorrect[0]}/{n_total[0]} split nodes INCORRECT "
          f"(already-pure node, or cuts within dominant frame-sig)")


if __name__ == "__main__":
    main()
