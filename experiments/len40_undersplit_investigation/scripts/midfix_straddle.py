"""Correct-level test of 'worse (less decisive) family -> better DFA'.

The DT splits a prefix by the family-mean of s + m + v over discovered MIDFIXES m,
not the raw s.  So fragmentation lives at the deeper midfix nodes.  We take round 0's
ACTUAL splitting midfixes and, for SINK-sig strings (round 0 shattered SINK across 4
states; round 1 kept it as 1), measure at each midfix m:
    - under ROUND 0's family+band: %decisive-accept / %reject / %indecisive, straddle
    - under ROUND 1's family+band on the SAME s+m contexts: same breakdown
Prediction if the mechanism holds: at the midfixes where round 0 decisively straddles
SINK (fuelling the split), round 1's family is more indecisive (abstains -> no split).
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
NPER = 150
NSUF = 40
DUMP = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/mr_dumps/seed2"


def sig(s):
    wc = 0; f0 = f1 = False
    for c in s:
        if c >= 3: wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return "SINK" if (f0 and f1) else (f0, f1, wc % 3)


def collect_midfixes(root):
    """All internal-node midfixes of a MidfixTree, from its raw (midfix, lookup) root."""
    out = []
    def rec(node):
        if not isinstance(node, tuple):
            return
        midfix, lookup = node
        out.append(bytes(midfix))
        rec(lookup[True]); rec(lookup[False])
    rec(root)
    return out


def main():
    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
    lifted = LiftedOracle(base, vocab, seed=SEED)

    r0 = pickle.load(open(f"{DUMP}/round_00.pkl", "rb"))
    r1 = pickle.load(open(f"{DUMP}/round_01.pkl", "rb"))
    fam0 = [bytes(v) for v in r0["dt"].base_family]
    fam1 = [bytes(v) for v in r1["dt"].base_family]
    v0 = [fam0[i] for i in np.random.default_rng(7).choice(len(fam0), min(NSUF, len(fam0)), replace=False)]
    v1 = [fam1[i] for i in np.random.default_rng(7).choice(len(fam1), min(NSUF, len(fam1)), replace=False)]
    at0, rt0 = r0["accept_thresh"], r0["reject_thresh"]
    at1, rt1 = r1["accept_thresh"], r1["reject_thresh"]

    midfixes = collect_midfixes(r0["dt"].root)
    # dedup, keep non-empty (empty = root, already known one-sided), cap for cost
    seen = set(); mids = []
    for m in midfixes:
        if m and m not in seen:
            seen.add(m); mids.append(m)
    print(f"round-0 tree internal midfixes: {len(midfixes)} ({len(mids)} distinct non-empty)", flush=True)

    # SINK-sig string pool
    samp = SuperSampler(vocab, 36); rng = np.random.default_rng(999)
    sinks = []
    while len(sinks) < NPER:
        w = list(samp.sample(rng, vocab.alphabet_size))
        if sig(w) == "SINK":
            sinks.append(w)

    def breakdown(strs, mid, fam, at, rt):
        combos = [bytes(s) + mid + v for s in strs for v in fam]
        lab = np.asarray(lifted.membership_queries(combos), float).reshape(len(strs), len(fam))
        m = lab.mean(1)
        pa = np.mean(m >= at); pr = np.mean(m < rt); pi = np.mean((m >= rt) & (m < at))
        dec = pa + pr
        return pa, pr, pi, (min(pa, pr) / dec if dec > 0 else 0.0)

    print("\nSINK strings at round-0's splitting midfixes (len m):")
    print("  midfix        | R0 fam: %A %R %ind strad | R1 fam: %A %R %ind strad")
    for mid in mids[:12]:
        a0, r0p, i0, s0 = breakdown(sinks, mid, v0, at0, rt0)
        a1, r1p, i1, s1 = breakdown(sinks, mid, v1, at1, rt1)
        print(f"  len{len(mid):<3}         | {a0*100:4.0f}{r0p*100:4.0f}{i0*100:5.0f} {s0:5.2f} "
              f"| {a1*100:4.0f}{r1p*100:4.0f}{i1*100:5.0f} {s1:5.2f}", flush=True)


if __name__ == "__main__":
    main()
