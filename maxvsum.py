"""Per round: which true states are cut backwards, how big each is, max vs sum."""
import contextlib, io, json, sys
import numpy as np
import orthogonal_dfa.l_star.cluster as cluster
from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliRegex
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle
import tests.lstar_common as lc

task, seed = sys.argv[1], int(sys.argv[2])
if task == "subseq":
    INNER, sig, noise, sampler = BernoulliRegex(regex=r".*1010101.*"), 0.3, None, None
else:
    import importlib.util
    spec = importlib.util.spec_from_file_location("nm", "near_miss_repro.py")
    nm = importlib.util.module_from_spec(spec); spec.loader.exec_module(nm)
    INNER, sig, noise, sampler = nm.INNER, 0.275, dict(p_0=0.15, p_1=0.7), nm.NearMissSampler(share=0.5)
TARGET = INNER.target_dfa()


def state_of(prefix):
    q = TARGET.initial_state
    for c in bytes(prefix):
        q = TARGET.transitions[q][c]
    return q

rows, orig = [], cluster.judge_family

def judge(pst, gate, v, vs, family_size, *a, **k):
    j = orig(pst, gate, v, vs, family_size, *a, **k)
    if j.reason != "undersized":
        rep = pst.table.representative
        pre = [p for p, keep in zip(pst.table.prefixes, rep) if keep]
        dec = pst.compute_decision(j.vs, rep)
        st = np.array([state_of(p) for p in pre])
        acc, rej = pst.accept_thresh, pst.reject_thresh
        decided = (dec >= acc) | (dec < rej)
        n_dec = int(decided.sum())
        wrong = []
        for q in sorted(set(st.tolist())):
            m = (st == q) & decided
            if not m.sum(): continue
            called_accept = (dec[m] >= acc).sum() >= (dec[m] < rej).sum()
            if called_accept != (q in TARGET.final_states):
                wrong.append((int(q), int(m.sum())))
        if wrong and n_dec:
            rows.append(dict(states=wrong, n_decided=n_dec,
                             mx=max(n for _, n in wrong) / n_dec,
                             total=sum(n for _, n in wrong) / n_dec))
    return j

cluster.judge_family = judge
try:
    with contextlib.redirect_stdout(io.StringIO()):
        learn_dfa(lambda nm_, s: NoisyOracle(INNER, nm_, s), min_signal_strength=sig, seed=seed,
                  **({"noise_model": AsymmetricBernoulli(**noise)} if noise else {}),
                  **({"sampler": sampler} if sampler else {}))
except Exception as e:
    print(json.dumps(dict(task=task, seed=seed, error=type(e).__name__))); raise SystemExit
cluster.judge_family = orig
print(json.dumps(dict(task=task, seed=seed, rounds=rows)))
