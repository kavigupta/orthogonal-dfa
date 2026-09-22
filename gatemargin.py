"""How close does the accept-preserving gate come to refusing, on real targets?

The gate passes a side while its share reading as its own class clears the band's
threshold.  Rewritten as a coverage error -- the share of that side belonging to
the other class -- it passes while that stays under (1 - eps/signal)/2.
"""
import contextlib, io, json, sys
import numpy as np
import orthogonal_dfa.l_star.cluster as cluster
from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliParityOracle, BernoulliRegex
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle

SPEC = {
 "modulo":      (lambda: BernoulliParityOracle(modulo=9, allowed_moduluses=(3,6)), 0.3, None),
 "subseq":      (lambda: BernoulliRegex(regex=r".*1010101.*"), 0.3, None),
 "two_subseq":  (lambda: BernoulliRegex(regex=r".*1111.*1111.*"), 0.3, None),
 "modulo_hard": (lambda: BernoulliParityOracle(modulo=9, allowed_moduluses=(3,6)), 0.2, None),
 "modulo_asym": (lambda: BernoulliParityOracle(modulo=9, allowed_moduluses=(3,6)), 0.15,
                 dict(p_0=0.10, p_1=0.40)),
}
task, seed = sys.argv[1], int(sys.argv[2])
mk, sig, noise = SPEC[task]
inner = mk()
etaIn, etaOut = (1 - noise["p_1"], noise["p_0"]) if noise else (0.5 - sig, 0.5 - sig)
gap = 1 - etaIn - etaOut
rows, orig = [], cluster.drift_verdict

def spy(pst, counts, *a, **k):
    v = orig(pst, counts, *a, **k)
    (ha, na), (hr, nr) = counts
    # share of each side belonging to the other class, from its read rate
    if na:
        rows.append(("accept", na, ((1 - etaIn) - ha / na) / gap, v))
    if nr:
        rows.append(("reject", nr, (hr / nr - etaOut) / gap, v))
    return v

cluster.drift_verdict = spy
status = "ok"
try:
    with contextlib.redirect_stdout(io.StringIO()):
        learn_dfa(lambda nm, s: NoisyOracle(inner, nm, s), min_signal_strength=sig,
                  seed=seed, **(dict(noise_model=AsymmetricBernoulli(**noise)) if noise else {}))
except Exception as e:
    status = f"{type(e).__name__}"
cluster.drift_verdict = orig
es = [e for _, n, e, _ in rows if n >= 30]
print(json.dumps(dict(task=task, seed=seed, status=status, tests=len(rows),
                      refused=sum(1 for *_, v in rows if v != "admitted"),
                      worst=round(max(es), 4) if es else None,
                      median=round(float(np.median(es)), 4) if es else None)))
