"""eps_cov as the theorem defines it: mass under D_j of prefixes the returned
family DECIDES and decides WRONGLY.  Fresh draws, never the prefix table."""
import contextlib, io, sys
import numpy as np
import orthogonal_dfa.l_star.cluster as cluster

N_FRESH = 4000


def measure(inner, sampler, noise, signal, seed, n_fresh=N_FRESH):
    from orthogonal_dfa.l_star.learn import learn_dfa
    from orthogonal_dfa.l_star.structures import NoisyOracle

    target = inner.target_dfa()
    rounds = []
    original = cluster.judge_family

    def judge(pst, gate, v, vs, family_size, *a, **k):
        judged = original(pst, gate, v, vs, family_size, *a, **k)
        if judged.reason == "undersized":
            return judged
        suffixes = [bytes(pst.table.suffix(i)) for i in judged.vs]
        rng = np.random.default_rng(0xC0FFEE + len(rounds))
        fresh = [sampler.sample(rng, inner.alphabet_size) for _ in range(n_fresh)]
        oracle = pst.oracle
        votes = np.array([
            np.mean(oracle.membership_queries([bytes(p) + s for s in suffixes]))
            for p in fresh
        ])
        truth = np.array([target.accepts_input(bytes(p)) for p in fresh])
        accept, reject = pst.accept_thresh, pst.reject_thresh
        called_accept, called_reject = votes >= accept, votes < reject
        decided = called_accept | called_reject
        wrong = (called_accept & ~truth) | (called_reject & truth)
        rounds.append((float(decided.mean()), float(wrong.mean())))
        return judged

    cluster.judge_family = judge
    status = "ok"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            learn_dfa(lambda nm, s: NoisyOracle(inner, nm, s),
                      min_signal_strength=signal, seed=seed,
                      noise_model=noise, sampler=sampler)
    except Exception as e:
        status = f"{type(e).__name__}: {str(e)[:60]}"
    finally:
        cluster.judge_family = original
    return status, rounds
