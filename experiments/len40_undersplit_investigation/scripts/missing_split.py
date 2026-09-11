"""Find a specific split round 0 MISSES (an undersplit: one state conflating SINK &
live) and explain why round 0 misses it while round 1 makes it.

1. walk a fresh eval set through round 0's DFA; the state with the largest
   min(#SINK,#live) is a missing split -- two frame classes sharing one state.
2. confirm round 1 separates those same strings (they reach different round-1 states).
3. at each of round 1's internal midfixes m, measure the family-decision on the
   conflated SINK-subset vs live-subset under round 0's family+band and round 1's
   family+band.  The split round 0 misses is an m where round 1's family+band puts
   SINK and live on OPPOSITE sides (splittable) but round 0's puts them on the SAME
   side (not splittable).  That m, and why the two bands read it differently, is the
   answer.
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
NSK = 60          # conflated SINK/live members sampled from the target state
NSUF = 40         # family suffixes sampled per round
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


def collect_midfixes(root):
    out = []
    def rec(node):
        if not isinstance(node, tuple):
            return
        midfix, lookup = node
        out.append(bytes(midfix))
        rec(lookup[True]); rec(lookup[False])
    rec(root)
    return out


def walk(dfa, w):
    s = dfa.initial_state
    for c in w:
        s = dfa.transitions[s][c]
    return s


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
    v0 = [fam0[i] for i in np.random.default_rng(7).choice(len(fam0), min(NSUF, len(fam0)), replace=False)]
    v1 = [fam1[i] for i in np.random.default_rng(7).choice(len(fam1), min(NSUF, len(fam1)), replace=False)]
    at0, rt0 = r0["accept_thresh"], r0["reject_thresh"]
    at1, rt1 = r1["accept_thresh"], r1["reject_thresh"]
    print(f"round 0 band [{rt0:.3f},{at0:.3f}) | round 1 band [{rt1:.3f},{at1:.3f})", flush=True)

    samp = SuperSampler(vocab, 36); rng = np.random.default_rng(SEED + 999)
    supers = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(NEVAL)]
    sk = [sig(w) for w in supers]
    st0 = [walk(dfa0, w) for w in supers]

    # step 1: most-conflated round-0 state
    by = collections.defaultdict(lambda: {"SINK": [], "live": []})
    for w, g, s in zip(supers, sk, st0):
        by[s][g].append(w)
    scored = sorted(by.items(), key=lambda kv: -min(len(kv[1]["SINK"]), len(kv[1]["live"])))
    S, groups = scored[0]
    nS, nL = len(groups["SINK"]), len(groups["live"])
    acc0 = getattr(dfa0, "final_states", set())
    print(f"\n[step1] most-conflated round-0 state s{S}: {nS} SINK + {nL} live "
          f"(round-0 label: {'ACCEPT' if S in acc0 else 'reject'})", flush=True)
    print("  round-0 state SINK/live purity table (top 6 by conflation):")
    for s, gr in scored[:6]:
        a, b = len(gr["SINK"]), len(gr["live"])
        print(f"    s{s:<3} {a:4d} SINK {b:4d} live  {'ACCEPT' if s in acc0 else 'reject'}")

    S_sink = [groups["SINK"][i] for i in np.random.default_rng(1).choice(nS, min(NSK, nS), replace=False)]
    S_live = [groups["live"][i] for i in np.random.default_rng(2).choice(nL, min(NSK, nL), replace=False)]

    # step 2: does round 1 separate them?
    r1_sink = collections.Counter(walk(dfa1, w) for w in S_sink)
    r1_live = collections.Counter(walk(dfa1, w) for w in S_live)
    print(f"\n[step2] where round 1 sends this state's members:")
    print(f"    SINK -> {dict(r1_sink)}")
    print(f"    live -> {dict(r1_live)}")

    # step 3: at each round-1 midfix, family decision on SINK-set vs live-set under each band
    mids = []
    seen = set()
    for m in collect_midfixes(dt1.root):
        if m not in seen:
            seen.add(m); mids.append(m)

    def decide_breakdown(strs, mid, fam, at, rt):
        combos = [bytes(s) + mid + v for s in strs for v in fam]
        lab = np.asarray(oracle.membership_queries(combos), float).reshape(len(strs), len(fam))
        mean = lab.mean(1)
        return (mean.mean(), np.mean(mean >= at) * 100, np.mean(mean < rt) * 100,
                np.mean((mean >= rt) & (mean < at)) * 100)

    print(f"\n[step3] at each round-1 midfix m: family decision on the conflated SINK vs live subsets")
    print(f"        (mean | %Accept %Reject %Indec).  A MISSING SPLIT = round1 separates "
          f"(SINK->reject, live->accept) where round0 does not.")
    print(f"\n  m(len) | round-0 fam+band                       | round-1 fam+band")
    print(f"         |   SINK: mean %A %R %I   live: mean %A %R %I |   SINK: mean %A %R %I   live: mean %A %R %I  | SPLIT?")
    for mid in ([b""] + mids)[:10]:
        s0 = decide_breakdown(S_sink, mid, v0, at0, rt0)
        l0 = decide_breakdown(S_live, mid, v0, at0, rt0)
        s1 = decide_breakdown(S_sink, mid, v1, at1, rt1)
        l1 = decide_breakdown(S_live, mid, v1, at1, rt1)
        # "split" heuristic: under round1, SINK majority reject & live majority accept (opposite sides)
        split1 = s1[2] >= 50 and l1[1] >= 50
        split0 = s0[2] >= 50 and l0[1] >= 50
        flag = ("round1-ONLY" if split1 and not split0 else
                "both" if split1 and split0 else
                "round0-only" if split0 else "neither")
        print(f"  len{len(mid):<3} | S:{s0[0]:.2f} {s0[1]:3.0f} {s0[2]:3.0f} {s0[3]:3.0f}  "
              f"L:{l0[0]:.2f} {l0[1]:3.0f} {l0[2]:3.0f} {l0[3]:3.0f} | "
              f"S:{s1[0]:.2f} {s1[1]:3.0f} {s1[2]:3.0f} {s1[3]:3.0f}  "
              f"L:{l1[0]:.2f} {l1[1]:3.0f} {l1[2]:3.0f} {l1[3]:3.0f} | {flag}", flush=True)


if __name__ == "__main__":
    main()
