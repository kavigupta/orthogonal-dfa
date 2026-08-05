"""How well E-L* recovers the oracle's prefix structure as a DFA -- experiment 2, the
counterpart to :mod:`prefix_information` (experiment 1, the ceiling on what is there).

Runs E-L* on an oracle and caches **every round's full state** (DFA, decision tree,
prefix x suffix masks), not just the topline agreement, so a round's DFA can be
inspected offline.  Then measures the learned DFA's agreement with the oracle on a
held-out set: as E-L* wired it (from its initial state), and after re-rooting at the
best start state -- the findings' reachability wall makes the initial state a
self-looping sink, so re-rooting recovers part of the signal E-L* discovered but could
not connect.  The realistic runs are slow (hours per round), hence permacached; each
round is also written to ``runs/<key>/`` so a crashed run keeps its finished rounds.
"""

import os
import pickle

import numpy as np
from permacache import permacache, stable_hash

from orthogonal_dfa.analysis.prefix_information import LABELS, _build_oracle
from orthogonal_dfa.l_star.lstar import counterexample_driven_synthesis
from orthogonal_dfa.l_star.preconditions import _endpoint
from orthogonal_dfa.l_star.prefix_suffix_tracker import (
    PrefixSuffixTracker,
    SearchConfig,
)
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.statistics import (
    compute_suffix_size_counterexample_gen,
    population_size_and_evidence_margin,
)

ORACLES = ["spliceai", "composition_residual"]
RUNS_DIR = "runs"


def _search_config(mss, addtl, fnr_limit):
    n, eps = population_size_and_evidence_margin(
        signal_strength=mss, acceptable_fpr=0.01, acceptable_fnr=0.01
    )
    return SearchConfig(
        suffix_family_size=n,
        evidence_margin=eps,
        decision_rule_fpr=0.01,
        suffix_size_counterexample_gen=compute_suffix_size_counterexample_gen(
            0.01, 0.5 + mss
        ),
        min_signal_strength=mss,
        num_addtl_prefixes=addtl,
        fnr_limit=fnr_limit,
    )


def _round_state(dfa, dt, pst):
    """The full per-round state -- same schema the driver pickles, plus the DFA/DT."""
    tbl = pst.table
    masks = (
        np.array(tbl._masks, dtype=np.int8)  # pylint: disable=protected-access
        if tbl._masks  # pylint: disable=protected-access
        else np.zeros((0, tbl.num_prefixes), dtype=np.int8)
    )
    return dict(
        dfa=dfa,
        dt=dt,
        num_states=dt.num_states,
        final_states=sorted(dfa.final_states),
        prefixes=[list(p) for p in tbl.prefixes],
        suffixes=[list(s) for s in tbl._suffixes],  # pylint: disable=protected-access
        masks=masks,
    )


@permacache(
    "orthogonal_dfa/analysis/elstar_learnability/rounds_v1",
    key_function=dict(
        run_dir=lambda _: None
    ),  # where round pickles go; not part of key
)
def elstar_rounds(
    oracle_name,
    *,
    length=95,
    mss=0.06,
    num_prefixes=500,
    addtl=300,
    fnr_limit=0.05,
    acc_threshold=0.98,
    max_rounds=8,
    seed=0,
    run_dir=RUNS_DIR,
):
    """Run E-L* on ``oracle_name`` and return every round's :func:`_round_state`.

    Each round is also written to ``run_dir/<key>/round_NN.pkl`` as it completes, so a
    run that dies mid-way keeps the rounds it finished."""
    oracle = _build_oracle(oracle_name, length)
    config = _search_config(mss, addtl, fnr_limit)
    pst = PrefixSuffixTracker.create(
        UniformSampler(length),
        np.random.default_rng(seed),
        oracle,
        config,
        num_prefixes=num_prefixes,
    )
    key = stable_hash(
        (oracle_name, length, mss, num_prefixes, addtl, acc_threshold, seed)
    )
    out = os.path.join(run_dir, key[:16])
    os.makedirs(out, exist_ok=True)
    rounds = []
    gen = counterexample_driven_synthesis(
        pst, additional_counterexamples=addtl, acc_threshold=acc_threshold
    )
    for i, (dfa, dt, pst_copy) in enumerate(gen):
        state = _round_state(dfa, dt, pst)
        rounds.append(state)
        with open(os.path.join(out, f"round_{i:02d}.pkl"), "wb") as f:
            pickle.dump(state, f)
        if pst_copy is None or i + 1 >= max_rounds:
            break
    return rounds


def _agreement(dfa, strings, truth, start=None):
    pred = np.array(
        [_endpoint(dfa, s, start) in dfa.final_states for s in strings], dtype=bool
    )
    return float((pred == truth).mean())


def _best_reroot_agreement(dfa, val, val_truth, test, test_truth):
    """Pick the start state that maximizes validation agreement, report it on test."""
    best_state = max(dfa.states, key=lambda q: _agreement(dfa, val, val_truth, q))
    return _agreement(dfa, test, test_truth, best_state)


@permacache(
    "orthogonal_dfa/analysis/elstar_learnability/measure_v1",
    key_function=dict(run_dir=lambda _: None),
)
def learnability(
    oracle_name,
    *,
    length=95,
    eval_count=8000,
    eval_seed=999,
    run_dir=RUNS_DIR,
    **run_kw,
):
    """Agreement of the final learned DFA with the oracle: base rate (best trivial DFA),
    as E-L* wired it, and after re-rooting (chosen on validation, scored on disjoint test).
    """
    rounds = elstar_rounds(oracle_name, length=length, run_dir=run_dir, **run_kw)
    dfa = rounds[-1]["dfa"]
    rng = np.random.default_rng(eval_seed)
    strings = rng.integers(0, 4, (eval_count, length)).tolist()
    truth = _build_oracle(oracle_name, length).membership_queries(strings)
    half = eval_count // 2
    val, test = strings[:half], strings[half:]
    val_truth, test_truth = truth[:half], truth[half:]
    accept = float(truth.mean())
    return dict(
        base_rate=max(accept, 1 - accept),
        learned=_agreement(dfa, test, test_truth),
        rerooted=_best_reroot_agreement(dfa, val, val_truth, test, test_truth),
        accept_rate=accept,
        num_states=rounds[-1]["num_states"],
    )


def learnability_bars(**kw):
    """:func:`learnability` for each oracle in :data:`ORACLES` (each permacached)."""
    return {name: learnability(name, **kw) for name in ORACLES}


def plot_learnability(results, ax=None):
    """Grouped bars per oracle: E-L* as-learned and best re-rooted agreement, over the
    base-rate line (what a trivial accept/reject DFA scores)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 3.5))
    names = list(results)
    x = np.arange(len(names))
    ax.bar(x - 0.2, [results[n]["learned"] for n in names], 0.4, label="as learned")
    ax.bar(x + 0.2, [results[n]["rerooted"] for n in names], 0.4, label="re-rooted")
    base = float(np.mean([results[n]["base_rate"] for n in names]))
    ax.axhline(base, linestyle="--", color="0.4", linewidth=1)
    ax.text(len(names) - 0.5, base, "base rate", va="bottom", ha="right", color="0.4")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(n, n) for n in names])
    ax.set_ylabel("DFA-vs-oracle agreement")
    ax.set_ylim(0, 1)
    ax.legend()
    return ax
