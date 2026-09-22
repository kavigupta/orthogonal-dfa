"""Mass of wrongly-cut states in the families rounds actually accept."""
import contextlib, io, json, os, sys
root, task, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, root); os.chdir(root)
from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliParityOracle, BernoulliRegex
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle
import tests.lstar_common as lc

SPEC = {
  "subseq_asym": (lambda: BernoulliRegex(regex=r".*1010101.*"), 0.2, AsymmetricBernoulli(p_0=0.15, p_1=0.7)),
  "modulo_asym": (lambda: BernoulliParityOracle(modulo=9, allowed_moduluses=(3,6)), 0.15, AsymmetricBernoulli(p_0=0.10, p_1=0.40)),
  "modulo_hard": (lambda: BernoulliParityOracle(modulo=9, allowed_moduluses=(3,6)), 0.2, None),
  "subseq":      (lambda: BernoulliRegex(regex=r".*1010101.*"), 0.3, None),
  "two_subseq":  (lambda: BernoulliRegex(regex=r".*1111.*1111.*"), 0.3, None),
}
inner_f, signal, noise = SPEC[task]; inner = inner_f()
creator = lambda nm, s: NoisyOracle(inner, nm, s)
kw = dict(noise_model=noise) if noise else {}
tracker = lc.RecordingTracker()
out = dict(task=task, seed=seed)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        lc.learn_dfa(creator, tracker=tracker, min_signal_strength=signal, seed=seed, **kw)
    truth = creator(lc.SymmetricBernoulli(p_correct=1.0), 0).target_dfa()
    rounds = []
    for c in tracker.classifiers:
        cuts = lc._state_cuts(c, truth)
        tot = sum(a + r for a, r in cuts.values())
        wrong = [(q, a + r) for q, (a, r) in cuts.items()
                 if (a >= r) != (q in truth.final_states)]
        rounds.append(dict(total=tot, n_states=len(cuts),
                           wrong=[(int(q), n, round(n/max(1,tot), 4)) for q, n in wrong],
                           worst_mass=round(max([n/max(1,tot) for _, n in wrong], default=0.0), 4)))
    out["rounds"] = rounds
except Exception as e:
    out["error"] = f"{type(e).__name__}: {str(e)[:120]}"
print(json.dumps(out))
