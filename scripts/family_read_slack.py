"""Family-read slack plots: collect (runs the learner, writes CSVs) and plot (reads CSVs only).

  python family_read_slack.py collect --targets bench:0 modulo ... --learn-seed 0 \
      --read-seed 1 --signal 0.3 --strings-per-state 1000 --workers 2 \
      --out OUT
  python family_read_slack.py plot --out OUT
  python family_read_slack.py all  <collect args>
"""

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KAPPA = 3.5e-3
EPSILON = 0.02  # 1 - acc_threshold
ACC_THRESHOLD = 0.98
CLEAN_U, BAND_U = 1e-3, 0.5
STRAY = (0.002, 0.009)

FAMILY_FIELDS = [
    "target",
    "round",
    "state",
    "accepting",
    "f",
    "m",
    "k",
    "exp_share",
    "accept_thresh",
    "reject_thresh",
    "center",
    "margin",
    "x_norm",
    "mass_end",
    "mass_walk",
    "length",
    "n",
    "n_correct",
    "n_wrong",
    "n_undecided",
    "u_emp",
    "wrong_emp",
    "u_th",
    "wrong_th",
    "correct_th",
    "eps_w",
    "fnr_limit",
    "good",
    "theta",
    "ceil_theta",
    "nQ",
]
ROUND_FIELDS = [
    "target",
    "round",
    "m",
    "accept_thresh",
    "reject_thresh",
    "fnr_limit",
    "eps_w",
    "ending",
    "probes",
    "n_start",
    "n_end",
    "n_stuck",
    "n_middle",
    "n_member",
    "n_edge",
    "n_agree",
    "final_probes",
    "final_ends",
    "final_middles",
    "whole_ends_share",
    "whole_union_share",
    "final_union_share",
    "ends_band",
    "ends_edge",
    "ends_clean",
    "pop_undecided_max",
    "pop_undecided_json",
    "guarantee_pop",
    "nQ",
    "theta",
    "ceil_theta",
    "consistency",
]


# ----------------------------------------------------------------------------- collect


def _build_target(spec):
    from orthogonal_dfa.l_star.examples.benchmark_generator import (
        DFAOracle,
        sample_balanced_benchmark,
    )
    from orthogonal_dfa.l_star.examples.bernoulli_parity import (
        BernoulliParityOracle,
        BernoulliRegex,
    )
    from orthogonal_dfa.l_star.structures import NoisyOracle

    if spec.startswith("bench:"):
        outer, _, _ = sample_balanced_benchmark(
            int(spec[6:]),
            alphabet_size=2,
            num_inner_states=12,
            num_outer_states=10,
            probe_length=40,
        )
        inner = DFAOracle(outer)
    elif spec == "modulo":
        inner = BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6))
    elif spec == "subseq":
        inner = BernoulliRegex(regex=r".*1010101.*")
    elif spec == "overlap":
        inner = BernoulliRegex(regex=r".*11011.*")
    elif spec == "two_subseq":
        inner = BernoulliRegex(regex=r".*1111.*1111.*")
    elif spec == "parity_toggle":
        from tests.e2e.test_walk_parity import build_parity_toggle

        inner = DFAOracle(build_parity_toggle())
    else:
        raise ValueError(spec)
    return inner, (lambda nm, s, _i=inner: NoisyOracle(_i, nm, s))


def _classify(u):
    return "clean" if u < CLEAN_U else ("edge" if u <= BAND_U else "band")


def _vote_law(m, k, p1, p0, accept_thresh, reject_thresh):
    """(P(accept), P(reject), P(undecided)) of the vote mean: k accepting suffixes read
    at rate p1, m-k rejecting ones at rate p0, summed exactly."""
    import numpy as np
    from scipy.stats import binom

    pmf = np.convolve(
        binom.pmf(np.arange(k + 1), k, p1), binom.pmf(np.arange(m - k + 1), m - k, p0)
    )
    means = np.array([c / m for c in range(m + 1)])
    acc = pmf[means >= accept_thresh].sum()
    rej = pmf[means < reject_thresh].sum()
    # tails computed separately for precision at 1e-10 and below
    k_acc = np.flatnonzero(means >= accept_thresh)
    k_rej = np.flatnonzero(means < reject_thresh)
    acc_tail = float(pmf[k_acc].sum()) if len(k_acc) else 0.0
    rej_tail = float(pmf[k_rej].sum()) if len(k_rej) else 0.0
    return acc_tail, rej_tail, max(0.0, 1 - acc_tail - rej_tail)


def _run_target(args):
    (spec, learn_seed, read_seed, signal_s, n_per) = args
    import numpy as np

    inner, creator = _build_target(spec)

    from orthogonal_dfa.l_star import counterexample_synthesis as cs
    from orthogonal_dfa.l_star.cluster import read_rates
    from orthogonal_dfa.l_star.counterexample_search import CounterexampleSearch
    from orthogonal_dfa.l_star.dfa_utils import (
        count_paths_to_state,
        sample_string_reaching_state,
    )
    from orthogonal_dfa.l_star.learn import build_pst
    from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, SymmetricBernoulli
    from orthogonal_dfa.l_star.tracker import SynthesisTracker

    oracles = []

    def capturing(nm, s):
        o = creator(nm, s)
        oracles.append(o)
        return o

    # Target in minimal form: equivalent states share f(q), so they share a read.
    dfa = inner.target_dfa().minify()
    syms = sorted(dfa.input_symbols)
    states = sorted(dfa.states)
    nQ = len(states)
    theta = EPSILON / (864 * (nQ + 1))

    pst = build_pst(capturing, min_signal_strength=signal_s, seed=learn_seed)
    oracle = oracles[0]
    nm = oracle.noise_model
    if isinstance(nm, SymmetricBernoulli):
        p1, p0 = nm.p_correct, 1 - nm.p_correct
    else:
        assert isinstance(nm, AsymmetricBernoulli)
        p1, p0 = nm.p_1, nm.p_0
    weights = pst.sampler.symbol_weights(pst.alphabet_size)
    L = pst.sampler.length
    wsum = sum(weights)

    # visitation mass: law of the state after L symbols, and averaged over positions 1..L
    dist = {q: 0.0 for q in states}
    dist[dfa.initial_state] = 1.0
    walk = {q: 0.0 for q in states}
    for _ in range(L):
        nxt = {q: 0.0 for q in states}
        for q, pr in dist.items():
            for s in syms:
                nxt[dfa.transitions[q][s]] += pr * weights[s] / wsum
        dist = nxt
        for q in states:
            walk[q] += dist[q] / L
    mass_end = dict(dist)

    def run(q, string):
        for c in string:
            q = dfa.transitions[q][c]
        return q

    finals = set(dfa.final_states)

    rows, round_rows = [], []
    state = {"vs": None, "boundary": None, "cat": {}, "round_meta": None}
    searches = []

    class LoggingSearch(CounterexampleSearch):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.log = []
            searches.append(self)

        def probe(self, x):
            outcome, charged = super().probe(x)
            self.log.append(
                (
                    type(outcome).__name__,
                    getattr(outcome, "undecided", None),
                    getattr(outcome, "stuck", False),
                )
            )
            return outcome, charged

    cs.CounterexampleSearch = LoggingSearch

    class Tr(SynthesisTracker):
        def on_family_resolved(self, suffixes, boundary, round_index):
            state["vs"], state["boundary"] = list(suffixes), boundary

        def on_round_classified(self, classifier, round_index):
            vs = state["vs"]
            m = len(vs)
            acc_t, rej_t = classifier.accept_thresh, classifier.reject_thresh
            center, margin = state["boundary"], (acc_t - rej_t) / 2
            eps_w = read_rates(pst, state["boundary"])[0]
            fnr = pst.fnr_limit
            # truth of each suffix from each state: does delta(q, v) accept
            cat = {}
            for qi, q in enumerate(states):
                accv = np.array([run(q, v) in finals for v in vs])
                k = int(accv.sum())
                f = k / m
                ok = mass_end[q] > 0
                length = L
                counts = None
                if walk[q] <= 0:
                    continue
                # length at which q is reachable (parity-like targets)
                for length in range(L, max(L - 8, 0), -1):
                    counts = count_paths_to_state(dfa, q, length, weights)
                    if counts[length][dfa.initial_state] > 0:
                        break
                    counts = None
                if counts is None:
                    continue
                rng = np.random.default_rng([read_seed, round_index, q])
                seen = set()
                budget = 30 * n_per
                while len(seen) < n_per and budget > 0:
                    budget -= 1
                    seen.add(sample_string_reaching_state(dfa, counts, rng, weights))
                zs = sorted(seen)
                truth = accv.tolist()
                votes = np.empty(len(zs))
                for i, z in enumerate(zs):
                    votes[i] = (
                        sum(
                            nm.apply_noise(t, z + v, oracle.seed)
                            for t, v in zip(truth, vs)
                        )
                        / m
                    )
                if qi == 0 and round_index == 0:
                    chk = [zs[0] + v for v in vs[:50]]
                    ref = oracle.membership_queries(chk)
                    mine = [
                        nm.apply_noise(t, c, oracle.seed)
                        for t, c in zip(truth[:50], chk)
                    ]
                    assert list(ref) == mine, "noise replay differs from NoisyOracle"
                dec_acc = votes >= acc_t
                dec_rej = votes < rej_t
                und = ~(dec_acc | dec_rej)
                is_acc = q in finals
                n_correct = int((dec_acc if is_acc else dec_rej).sum())
                n_wrong = int((dec_rej if is_acc else dec_acc).sum())
                pa, pr_, pu = _vote_law(m, k, p1, p0, acc_t, rej_t)
                exp_share = f * p1 + (1 - f) * p0
                good = exp_share >= acc_t or exp_share < rej_t
                cat[q] = _classify(pu)
                rows.append(
                    dict(
                        target=spec,
                        round=round_index,
                        state=q,
                        accepting=int(is_acc),
                        f=f,
                        m=m,
                        k=k,
                        exp_share=exp_share,
                        accept_thresh=acc_t,
                        reject_thresh=rej_t,
                        center=center,
                        margin=margin,
                        x_norm=(exp_share - center) / margin,
                        mass_end=mass_end[q],
                        mass_walk=walk[q],
                        length=length,
                        n=len(zs),
                        n_correct=n_correct,
                        n_wrong=n_wrong,
                        n_undecided=int(und.sum()),
                        u_emp=und.mean(),
                        wrong_emp=n_wrong / len(zs),
                        u_th=pu,
                        wrong_th=(pr_ if is_acc else pa),
                        correct_th=(pa if is_acc else pr_),
                        eps_w=eps_w,
                        fnr_limit=fnr,
                        good=int(good),
                        theta=theta,
                        ceil_theta=1.5 * theta,
                        nQ=nQ,
                    )
                )
            state["cat"] = cat
            rep = pst.table.representative
            decision = pst.compute_decision(
                [pst.table.intern_suffix(v) for v in vs], rep
            )
            pops = {}
            for label, members in pst.table.population_masks().items():
                if members.sum():
                    pops[str(label)] = float(
                        (members & (decision >= rej_t) & (decision < acc_t)).sum()
                        / members.sum()
                    )
            state["round_meta"] = dict(
                m=m,
                accept_thresh=acc_t,
                reject_thresh=rej_t,
                fnr_limit=fnr,
                eps_w=eps_w,
                pops=pops,
            )

        def on_search_ended(self, ending, round_index):
            s = searches[-1]
            meta = state["round_meta"]
            kinds = {}
            cats = {"band": 0, "edge": 0, "clean": 0}
            for kind, und, stuck in s.log:
                name = "stuck" if (kind == "End" and stuck) else kind
                kinds[name] = kinds.get(name, 0) + 1
                if und is not None and kind in ("Start", "End"):
                    cats[state["cat"].get(run(dfa.initial_state, und), "clean")] += 1
            probes = max(s.probes, 1)
            n_ends = kinds.get("Start", 0) + kinds.get("End", 0) + kinds.get("stuck", 0)
            fp = max(s.counts.probes, 1)
            fe, fm = len(s.counts.ends), len(s.counts.middles)
            pu = meta["pops"]
            round_rows.append(
                dict(
                    target=spec,
                    round=round_index,
                    m=meta["m"],
                    accept_thresh=meta["accept_thresh"],
                    reject_thresh=meta["reject_thresh"],
                    fnr_limit=meta["fnr_limit"],
                    eps_w=meta["eps_w"],
                    ending=ending.kind,
                    probes=s.probes,
                    n_start=kinds.get("Start", 0),
                    n_end=kinds.get("End", 0),
                    n_stuck=kinds.get("stuck", 0),
                    n_middle=kinds.get("Middle", 0),
                    n_member=kinds.get("Member", 0),
                    n_edge=kinds.get("EdgeFound", 0),
                    n_agree=kinds.get("Agree", 0),
                    final_probes=s.counts.probes,
                    final_ends=fe,
                    final_middles=fm,
                    whole_ends_share=n_ends / probes,
                    whole_union_share=(n_ends + kinds.get("Middle", 0)) / probes,
                    final_union_share=(fe + fm) / fp,
                    ends_band=cats["band"],
                    ends_edge=cats["edge"],
                    ends_clean=cats["clean"],
                    pop_undecided_max=max(pu.values()) if pu else 0.0,
                    pop_undecided_json=json.dumps(pu),
                    guarantee_pop=2 * meta["fnr_limit"],
                    nQ=nQ,
                    theta=theta,
                    ceil_theta=1.5 * theta,
                    consistency="",
                )
            )

        def on_consistency_estimated(self, consistency, round_index):
            round_rows[round_index]["consistency"] = consistency

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        cs.do_counterexample_driven_synthesis(
            pst, acc_threshold=ACC_THRESHOLD, tracker=Tr()
        )
    return spec, rows, round_rows, time.time() - t0


def _write(path, fields, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def collect(a):
    os.makedirs(a.out, exist_ok=True)
    jobs = [
        (
            t,
            a.learn_seed,
            a.read_seed,
            a.signal,
            a.strings_per_state,
        )
        for t in a.targets
    ]
    fam, rnd = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for spec, rows, rrows, secs in ex.map(_run_target, jobs):
            print(
                f"{spec}: {len(rrows)} rounds, {len(rows)} state-rounds, {secs:.0f}s",
                flush=True,
            )
            fam += rows
            rnd += rrows
    _write(os.path.join(a.out, "family_reads.csv"), FAMILY_FIELDS, fam)
    _write(os.path.join(a.out, "round_level.csv"), ROUND_FIELDS, rnd)


# ----------------------------------------------------------------------------- plot


def _load(path):
    import pandas as pd

    return pd.read_csv(path)


def plot(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fam = _load(os.path.join(a.out, "family_reads.csv"))
    rnd = _load(os.path.join(a.out, "round_level.csv"))
    targets = list(dict.fromkeys(fam.target))
    pal = [
        "#2a6fbb",
        "#d9822b",
        "#3a9b6a",
        "#9b59b6",
        "#b8860b",
        "#c0392b",
        "#17a2b8",
        "#7f8c8d",
        "#e84393",
    ]
    col = {t: pal[i % len(pal)] for i, t in enumerate(targets)}
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    outdir = a.out

    def floor_of(df):
        return 0.5 / df.n

    # ---------------- A
    fig, ax = plt.subplots(figsize=(13, 8))
    fam = fam.copy()
    fam["u_plot"] = np.where(fam.u_emp > 0, fam.u_emp, floor_of(fam))
    zero = fam.u_emp == 0
    # theory curves per (target, round)
    # recover p1, p0 from the data: exp_share = f p1 + (1-f) p0 (two states suffice)
    sub = fam.drop_duplicates("f")
    A_ = np.vstack([sub.f, 1 - sub.f]).T
    p1, p0 = np.linalg.lstsq(A_, sub.exp_share, rcond=None)[0]
    for (t, r), g in fam.groupby(["target", "round"]):
        row = g.iloc[0]
        m = int(row.m)
        ks = np.arange(0, m + 1, max(1, m // 150))
        xs, us = [], []
        for k in ks:
            f = k / m
            _, _, u = _vote_law(m, int(k), p1, p0, row.accept_thresh, row.reject_thresh)
            xs.append((f * p1 + (1 - f) * p0 - row.center) / row.margin)
            us.append(max(u, 1e-12))
        ax.plot(xs, us, color="#999999", lw=0.6, alpha=0.5, zorder=1)
    ax.plot(
        [],
        [],
        color="#999999",
        lw=0.8,
        label="theory: vote law vs f (one line per round)",
    )
    size = 10 + 600 * fam.mass_walk.clip(upper=0.5)
    for t in targets:
        g = fam[fam.target == t]
        s = size[g.index]
        ax.scatter(
            g.x_norm,
            g.u_plot,
            s=s,
            color=col[t],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.5,
            label=t,
            zorder=3,
        )
    z = fam[zero]
    ax.scatter(
        z.x_norm,
        z.u_plot,
        s=18,
        marker="v",
        facecolor="none",
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
        label="zero undecided observed (plotted at 0.5/n)",
    )
    ax.axvline(-1, color="k", ls="--", lw=1)
    ax.axvline(1, color="k", ls="--", lw=1)
    ax.text(0, 2.0, "band", ha="center", va="bottom", fontsize=9)
    good = fam[fam.good == 1]
    wmed = np.median(good.u_th) if len(good) else np.nan

    def gap(level, color_ok, label, text_y=None):
        ax.axhline(level, color=color_ok, lw=1.4, zorder=2, label=label)

    lines = []
    for fl in sorted(set(rnd.guarantee_pop.round(6))):
        ax.axhline(fl, color="#c0392b", lw=1.2, zorder=2)
        lines.append(fl)
    ax.plot(
        [],
        [],
        color="#c0392b",
        lw=1.2,
        label="2*fnr_limit, per-population guarantee (0.2 down to 0.05)",
    )
    for cq in sorted(set(fam.ceil_theta)):
        ax.axhline(cq, color="#2a6fbb", lw=1.0, ls=":", zorder=2)
    ax.plot(
        [],
        [],
        color="#2a6fbb",
        lw=1.0,
        ls=":",
        label="1.5*theta good-state ceiling (one line per |Q|)",
    )
    ax.axhspan(
        *STRAY,
        color="#3a9b6a",
        alpha=0.18,
        zorder=0,
        label="measured stray undecided 0.2-0.9% per read",
    )
    # gap shading: from the median undecided rate at good states up to each line
    gu = good.u_emp[good.u_emp > 0]
    if len(gu):
        med = float(np.median(gu))
        top = max(lines)
        ax.axhspan(med, top, xmin=0, xmax=1, color="#c0392b", alpha=0.05, zorder=0)
        c15 = float(np.median(good.ceil_theta))
        lo, hi = sorted([med, c15])
        ax.axhspan(
            lo, hi, color="#d9822b" if med > c15 else "#2a6fbb", alpha=0.10, zorder=0
        )
        ax.annotate(
            f"median observed U at good states (nonzero) = {med:.1e}\n"
            f"gap to 2*fnr_limit: x{max(lines) / med:.0f} (proof slack); "
            f"to 1.5*theta: x{med / c15:.0f} (observed exceeds ceiling)",
            xy=(0.02, 0.04),
            xycoords="axes fraction",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#999999"),
        )
    # violators: good-state points above the ceiling
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 3)
    ax.set_xlabel(
        "expected vote share, normalised: (E[vote] - boundary) / margin   (band = [-1, 1])"
    )
    ax.set_ylabel("P(undecided) per state and round (log)")
    ax.set_title(
        "A. Family-read undecided rate per target state vs the read band\n"
        "(marker area = visitation mass; shaded: gap between observed and each guarantee)"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "A_undecided_vs_band.png"), dpi=130)
    plt.close(fig)

    # ---------------- B
    fig, ax = plt.subplots(figsize=(9, 7.5))
    b = fam.copy()
    b["w_plot"] = np.where(b.wrong_emp > 0, b.wrong_emp, floor_of(b))
    for t in targets:
        g = b[b.target == t]
        ax.scatter(
            g.u_th.clip(lower=1e-12),
            g.wrong_th.clip(lower=1e-30),
            s=14,
            color=col[t],
            alpha=0.6,
            label=t,
        )
    u = np.logspace(-12, 0, 100)
    epsw = float(fam.eps_w.median())
    ax.plot(u, KAPPA * u, "k--", lw=1, label="kappa*U (kappa=3.5e-3)")
    ax.plot(
        u, epsw + KAPPA * u, "k-", lw=1.4, label=f"eps_w + kappa*U (eps_w={epsw:.1e})"
    )
    ax.axhline(epsw, color="#c0392b", lw=1, ls=":", label="eps_w floor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-30, 1)
    ax.set_xlim(1e-12, 2)
    ax.set_xlabel("P(undecided) U (exact vote law)")
    ax.set_ylabel("P(wrong side) (exact vote law)")
    ratio = (fam.wrong_th - fam.eps_w) / fam.u_th.clip(lower=1e-300)
    ratio = ratio[fam.wrong_th > 0]
    viol = int(((fam.wrong_th > fam.eps_w + KAPPA * fam.u_th)).sum())
    ax.set_title(
        f"B. Wrong vs undecided per (state, round); points above the solid line violate the assumption\n"
        f"{viol} of {len(fam)} points above it; max (wrong-eps_w)/U = {ratio.max():.2e}",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "B_wrong_vs_undecided.png"), dpi=130)
    plt.close(fig)

    # ---------------- C
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    bins = np.logspace(-12, 0.05, 27)
    uu = fam.u_th.clip(lower=1e-12)
    axs[0].hist(uu, bins=bins, color="#2a6fbb")
    axs[0].set_title("C1. unweighted: (state, round) count")
    axs[1].hist(uu, bins=bins, weights=fam.mass_walk, color="#d9822b")
    axs[1].set_title("C2. weighted by visitation mass (mass_walk)")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("P(undecided) (exact vote law, floored at 1e-12)")
        ax.axvline(STRAY[0], color="#3a9b6a", ls=":")
        ax.axvline(STRAY[1], color="#3a9b6a", ls=":")
    axs[0].set_ylabel("count")
    axs[1].set_ylabel("summed mass")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "C_undecided_hist.png"), dpi=130)
    plt.close(fig)

    # ---------------- F slack summary
    def stats(vals):
        v = np.array([x for x in vals if np.isfinite(x) and x > 0])
        return (np.median(v), v.min(), v.max()) if len(v) else (np.nan,) * 3

    q1, q2, q3, q4 = {}, {}, {}, {}
    for t in targets:
        g = fam[fam.target == t]
        r = rnd[rnd.target == t]
        v1, v2 = [], []
        for rd, gg in g.groupby("round"):
            gd = gg[gg.good == 1]
            if len(gd) and gd.u_th.max() > 0:
                v1.append(gd.ceil_theta.iloc[0] / gd.u_th.max())
            gw = gg[gg.wrong_th > 0]
            if len(gw):
                v2.append(((gw.eps_w + KAPPA * gw.u_th) / gw.wrong_th).min())
        q1[t], q2[t] = stats(v1), stats(v2)
        q3[t] = stats(
            (r.guarantee_pop / r.pop_undecided_max.replace(0, np.nan)).tolist()
        )
        q4[t] = stats((EPSILON / r.final_union_share.replace(0, np.nan)).tolist())
    panels = [
        ("F1. good-state undecided: 1.5*theta / max_good U(q)", q1),
        ("F2. wrong rate: (eps_w + kappa*U) / wrong, worst state", q2),
        ("F3. per-population undecided: 2*fnr_limit / observed max", q3),
        (
            "F4. probes with undecided start/end/search: eps(0.02) / observed (final segment)",
            q4,
        ),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (title, qs) in zip(axs.ravel(), panels):
        xs = np.arange(len(targets))
        for i, t in enumerate(targets):
            med, lo, hi = qs[t]
            if not np.isfinite(med):
                continue
            c = "#c0392b" if med < 1 else "#2a6fbb"
            ax.plot([i, i], [lo, hi], color=c, lw=3, alpha=0.6)
            ax.plot(i, med, "o", color=c, ms=8, mec="white")
        ax.axhline(1, color="k", lw=1)
        ax.set_yscale("log")
        ax.axhspan(1e-12, 1, color="#c0392b", alpha=0.08)
        ax.axhspan(1, 1e12, color="#2a6fbb", alpha=0.05)
        ax.set_xticks(xs)
        ax.set_xticklabels(targets, rotation=35, ha="right")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("bound / observed (median, range over rounds)")
        allv = [x for t in targets for x in qs[t] if np.isfinite(x)]
        if allv:
            ax.set_ylim(min(allv + [0.5]) / 3, max(allv + [2]) * 3)
        ax.text(
            0.01,
            0.97,
            "above 1: proof too pessimistic (slack)",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="#2a6fbb",
        )
        ax.text(
            0.01,
            0.03,
            "below 1 (red): reality violates the bound",
            transform=ax.transAxes,
            va="bottom",
            fontsize=8,
            color="#c0392b",
        )
    fig.suptitle("F. Slack summary: proof bound divided by observed value", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "F_slack_summary.png"), dpi=130)
    plt.close(fig)

    # ---------------- E
    fig, ax = plt.subplots(figsize=(11, 5.5))
    grp = rnd.groupby("target")[
        ["ends_band", "ends_edge", "ends_clean", "probes"]
    ].sum()
    grp = grp.reindex(targets)
    bottom = np.zeros(len(grp))
    for key, c, lab in (
        ("ends_clean", "#3a9b6a", "clean state"),
        ("ends_edge", "#d9822b", "band-edge state"),
        ("ends_band", "#c0392b", "in-band state"),
    ):
        v = (grp[key] / grp.probes).to_numpy()
        ax.bar(grp.index, v, bottom=bottom, color=c, label=lab)
        bottom += v
    ax.axhline(EPSILON, color="k", ls="--", lw=1, label="epsilon = 0.02")
    ax.set_ylabel("share of probes whose start/end sift stops undecided (all rounds)")
    ax.set_title(
        "E. Probes ending undecided at start/end, by the class of the deciding read's state"
    )
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "E_probe_ends.png"), dpi=130)
    plt.close(fig)

    # ---------------- D (guarantee gaps: summary table plot)
    fig, ax = plt.subplots(figsize=(10, 5))
    names = ["1.5*theta ceiling", "2*fnr_limit (pop.)", "stray 0.2%", "stray 0.9%"]
    levels = [fam.ceil_theta.median(), rnd.guarantee_pop.median(), STRAY[0], STRAY[1]]
    ax.bar(names, levels, color=["#2a6fbb", "#c0392b", "#3a9b6a", "#3a9b6a"], alpha=0.6)
    gu = good.u_emp[good.u_emp > 0]
    obs = float(np.median(gu)) if len(gu) else np.nan
    ax.axhline(
        obs,
        color="k",
        lw=1.5,
        label=f"median nonzero observed U at good states = {obs:.1e}",
    )
    for i, lv in enumerate(levels):
        ax.text(i, lv * 1.3, f"x{lv / obs:.2g} of observed", ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("level")
    ax.set_title(
        "D. Guarantee levels against the observed good-state undecided rate (gap = ratio)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "D_guarantee_levels.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("collect", "all"):
        p = sub.add_parser(name)
        p.add_argument("--targets", nargs="+", required=True)
        p.add_argument("--learn-seed", type=int, required=True)
        p.add_argument("--read-seed", type=int, required=True)
        p.add_argument("--signal", type=float, required=True)
        p.add_argument("--strings-per-state", type=int, required=True)
        p.add_argument("--workers", type=int, required=True)
        p.add_argument("--out", required=True)
    p = sub.add_parser("plot")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd in ("collect", "all"):
        collect(a)
    if a.cmd in ("plot", "all"):
        plot(a)


if __name__ == "__main__":
    main()
