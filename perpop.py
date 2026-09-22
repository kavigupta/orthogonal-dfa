"""Per-population decided-and-wrong fraction -- the theorem's quantity, for each j."""
import contextlib, io, sys
import numpy as np
import orthogonal_dfa.l_star.cluster as cluster
from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliRegex
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle
import importlib.util
spec = importlib.util.spec_from_file_location("nm", "near_miss_repro.py")
nm = importlib.util.module_from_spec(spec); spec.loader.exec_module(nm)

share, seed = float(sys.argv[1]), int(sys.argv[2])
INNER = BernoulliRegex(regex=r".*1010101.*")
TARGET = INNER.target_dfa()
worst = []
original = cluster.judge_family

def judge(pst, gate, v, vs, family_size, *a, **k):
    j = original(pst, gate, v, vs, family_size, *a, **k)
    if j.reason != "undersized":
        rep = pst.table.representative
        pre = [p for p, keep in zip(pst.table.prefixes, rep) if keep]
        dec = pst.compute_decision(j.vs, rep)
        truth = np.array([TARGET.accepts_input(bytes(p)) for p in pre])
        acc, rej = pst.accept_thresh, pst.reject_thresh
        ca, cr = dec >= acc, dec < rej
        wrong = (ca & ~truth) | (cr & truth)
        for label, mask in pst.table.population_masks().items():
            n = int(mask.sum())
            if n:
                worst.append((str(label), n, float(wrong[mask].mean())))
    return j

cluster.judge_family = judge
status = "ok"
try:
    with contextlib.redirect_stdout(io.StringIO()):
        learn_dfa(lambda nm_, s: NoisyOracle(INNER, nm_, s), min_signal_strength=0.275,
                  seed=seed, noise_model=AsymmetricBernoulli(p_0=0.15, p_1=0.7),
                  sampler=nm.NearMissSampler(share=share))
except Exception as e:
    status = f"{type(e).__name__}: {str(e)[:60]}"
cluster.judge_family = original
import collections
by = collections.defaultdict(list)
for lbl, n, f in worst:
    by[lbl].append((n, f))
print(f"seed={seed} status={status}")
for lbl in sorted(by):
    rows = by[lbl]
    mx = max(f for _, f in rows)
    flag = "  <-- above eps_cov=0.05" if mx > 0.05 else ""
    print(f"    {lbl:<18} rounds={len(rows):<3} max decided-and-wrong {mx:.1%}{flag}")
