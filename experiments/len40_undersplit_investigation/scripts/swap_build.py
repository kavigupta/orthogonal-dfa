"""Controlled swap: is the DFA's frame-alignment governed by the POOL or the FAMILY?

Round 1 changed BOTH the representative pool (872 -> 1268, strict superset) and the
freshly-drawn suffix family.  Here we rebuild the DFA from every (pool, family) combo,
injecting each directly into a fresh pst (pool -> table, family -> interned suffix rows,
thresholds -> decision_boundary/evidence_margin taken from that family's own round).
Calibration is bypassed (thresholds set directly), so representative is calibration-only
and reconstructed as len>4 (only the short <=4 core is non-representative).

  baselines: (0,0) should reproduce round 0 (~15 states, low SINK/live homog),
             (1,1) should reproduce round 1 (~10 states, homog ~1.0).
  swaps:     (1,0) round-1 pool + round-0 family;  (0,1) round-0 pool + round-1 family.
If alignment follows the POOL -> pool causal (family not the lever).
If it follows the FAMILY -> family causal.
"""
import pickle, collections, math, time
import numpy as np
import scipy.stats  # noqa: F401  (module references scipy.stats)
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.counterexample_synthesis import (
    COUNTEREXAMPLE_PROBES, _default_patience)
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.lstar import estimate_agreement_rate
from orthogonal_dfa.l_star.mask_table import MaskTable
from orthogonal_dfa.l_star.transition_resolver import TransitionResolver
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

SEED = 2
ACC = 0.98
NEVAL = 3000
# Reduced from the real 4000 so a misaligned config's split-thrash stays tractable;
# applied UNIFORMLY to all four configs so the pool-vs-family comparison stays fair.
MAX_PROBES = 1200
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


def entropy(counts):
    n = sum(counts)
    return -sum((c / n) * math.log(c / n) for c in counts if c > 0) if n else 0.0


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

    def oracle_creator(_nm, s):
        return LiftedOracle(base, vocab, seed=s)

    samp = SuperSampler(vocab, 36); rng = np.random.default_rng(SEED + 999)
    supers = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(NEVAL)]
    sink = [sig(w) for w in supers]

    recs = {r: pickle.load(open(f"{DUMP}/round_{r:02d}.pkl", "rb")) for r in (0, 1)}
    for r in (0, 1):
        print(f"  round {r} baseline (dumped): {recs[r]['n_states']} states, "
              f"est {recs[r]['true_acc']:.3f}, boundary {recs[r]['boundary']:.3f}, "
              f"margin {recs[r]['evidence_margin']:.3f}, pool {len(recs[r]['prefixes'])}", flush=True)

    def walk(dfa, w):
        s = dfa.initial_state
        for c in w: s = dfa.transitions[s][c]
        return s

    def build(pool_r, fam_r):
        pst = build_pst(oracle_creator, min_signal_strength=0.05, seed=SEED,
                        sampler=SuperSampler(vocab, 36))
        pst.config.fnr_limit = 0.10
        pool = [bytes(p) for p in recs[pool_r]["prefixes"]]
        rep = [len(p) > 4 for p in pool]
        pst.table = MaskTable(pst.oracle, pool, rep)
        pst.decision_boundary = float(recs[fam_r]["boundary"])
        pst.evidence_margin = float(recs[fam_r]["evidence_margin"])
        fam = [bytes(v) for v in recs[fam_r]["dt"].base_family]
        vs = [pst.table.intern_suffix(v) for v in fam]
        t = time.time(); resolver = TransitionResolver(pst, vs)
        print(f"    [t] init+pool.add {time.time()-t:.0f}s", flush=True)
        t = time.time(); resolver.close_edges()
        print(f"    [t] close_edges {time.time()-t:.0f}s", flush=True)
        t = time.time()
        resolver.counterexample_pass(max_probes=MAX_PROBES, patience=_default_patience(ACC))
        print(f"    [t] ce_pass {time.time()-t:.0f}s", flush=True)
        t = time.time(); dfa, dt = resolver.to_dfa_and_tree()
        print(f"    [t] to_dfa {time.time()-t:.0f}s", flush=True)
        t = time.time()
        est = estimate_agreement_rate(pst, pst.sampler, pst.oracle, dt, dfa,
                                      num_samples=2000, acc_threshold=ACC)
        print(f"    [t] est {time.time()-t:.0f}s", flush=True)
        st = [walk(dfa, w) for w in supers]
        return len(dfa.states), homogeneity(st, sink), float(est), len(fam)

    print(f"\n(MAX_PROBES={MAX_PROBES}; baselines may not exactly reproduce 15/10 but "
          f"pool-vs-family DIRECTION is what matters)", flush=True)
    print("\n pool fam | states  SINK/live-homog   est    |famsize|  secs  tag", flush=True)
    # informative order: aligned baseline, both swaps, misaligned baseline last
    import os as _os
    _cfgs = [(1, 1)] if _os.environ.get("ONE") else [(1, 1), (0, 1), (1, 0), (0, 0)]
    for pr, fr in _cfgs:
        t0 = time.time()
        ns, h, e, fs = build(pr, fr)
        tag = "baseline" if pr == fr else "SWAP"
        print(f"   {pr}   {fr}  |  {ns:4d}      {h:.3f}         {e:.3f}    {fs:3d}   {time.time()-t0:5.0f}  {tag}", flush=True)


if __name__ == "__main__":
    main()
