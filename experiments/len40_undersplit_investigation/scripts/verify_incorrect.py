"""Verify which of round 0's distinguishers are INCORRECT: do they cut WITHIN a frame state
(resolve residual) or separate DIFFERENT frame states (legitimate)?

For each round-0 SPLIT_LOG distinguisher m: take a population of prefixes, group by fine
frame-sig (f0,f1,phase), append m, and compute the family accept-rate of prefix+m. A
distinguisher is INCORRECT if, within some single fine-sig group, it splits the members
(some >= accept, some < reject) -- i.e. it separates frame-equivalent prefixes. A CORRECT
distinguisher gives a consistent answer within each fine-sig group.
"""
import re, pickle, collections, os
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

SEED = 2; NPREF = 400; NSUF = 60
D = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/mr_dumps/seed2"
SYM = {0: 'TAG', 1: 'TAA', 2: 'TGA', 3: 'X', 4: 'Y'}


def finesig(s):
    wc = 0; f0 = f1 = False
    for c in s:
        if c >= 3: wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return "SINK" if (f0 and f1) else (f0, f1, wc % 3)


def main():
    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
    oracle = LiftedOracle(base, vocab, seed=SEED)
    ROUND = int(os.environ.get("ROUND", "0"))
    rec = pickle.load(open(f"{D}/round_{ROUND:02d}.pkl", "rb"))
    fam = [bytes(v) for v in rec["dt"].base_family]
    vs = [fam[i] for i in np.random.default_rng(7).choice(len(fam), min(NSUF, len(fam)), replace=False)]
    at, rt = rec["accept_thresh"], rec["reject_thresh"]

    # distinguishers from round 0's split log
    lines = open("../splitlog_s2.txt").read().strip().splitlines()
    dists = []; ri = -1
    for ln in lines:
        n = int(re.search(r'n=(\d+)', ln).group(1))
        if n == 2: ri += 1
        if ri == ROUND:
            d = re.search(r'distinguisher=\[([^\]]*)\]', ln).group(1)
            dists.append((n, [int(x) for x in d.split(',')] if d else []))

    # a mixed-length population, grouped by fine-sig
    prefs = []
    for L in (6, 12, 18, 24, 30):
        samp = SuperSampler(vocab, L); rng = np.random.default_rng(100 + L)
        prefs += [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(NPREF // 5)]
    groups = collections.defaultdict(list)
    for p in prefs:
        groups[finesig(p)].append(p)

    print(f"population {len(prefs)} prefixes; band [{rt:.3f},{at:.3f})\n")
    print("  n  distinguisher                      verdict   (within-frame-state split? = INCORRECT)")
    for n, m in dists:
        # for each fine-sig group, family accept-rate of p+m
        worst = None
        for g, ps in groups.items():
            if len(ps) < 15 or g == "SINK":
                continue
            combos = [bytes(p) + bytes(m) + v for p in ps for v in vs]
            mean = np.asarray(oracle.membership_queries(combos), float).reshape(len(ps), len(vs)).mean(1)
            acc = np.mean(mean >= at); rej = np.mean(mean < rt)
            # "split within group" = both a decisive-accept and decisive-reject fraction present
            if acc >= 0.2 and rej >= 0.2:
                if worst is None or min(acc, rej) > worst[1]:
                    worst = (g, min(acc, rej), acc, rej)
        ren = " ".join(SYM[x] for x in m) if m else "(empty)"
        if worst:
            g, mn, a, r = worst
            f0, f1, ph = g
            print(f" {n:2d}  [{ren:30}]  INCORRECT  cuts within (f0={int(f0)},f1={int(f1)},ph={ph}): "
                  f"{a*100:.0f}%A / {r*100:.0f}%R", flush=True)
        else:
            print(f" {n:2d}  [{ren:30}]  ok (consistent within every frame state)", flush=True)


if __name__ == "__main__":
    main()
