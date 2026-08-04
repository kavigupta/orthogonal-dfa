r"""Run E-L\* (counterexample-driven DFA synthesis) against the SpliceAI oracle.

Wraps the SpliceAI model (or a composition-residual / set-difference variant) as a
batched membership oracle and drives ``counterexample_driven_synthesis``. Saves a
pickle per round (prefix x suffix table + decision tree + DFA) for offline analysis,
and prints the REAL DFA-vs-oracle agreement each round on a fixed held-out set.

Because state discovery uses length-``sampler_len`` prefixes and transitions are
resolved by a majority vote over those prefixes, the learned DFA's initial state
tends to be a self-looping "bulk" sink, so raw agreement reads at the base rate;
the signal is recovered by re-rooting the saved DFA (see analysis in the writeup).

Examples::

    python scripts/run_elstar_spliceai.py --mss 0.1 --sampler-len 95
    python scripts/run_elstar_spliceai.py --residual-perlen --mss 0.06 --max-rounds 12
    python scripts/run_elstar_spliceai.py --setdiff resid           # SpliceAI \\ FM
"""

import argparse
import os
import pickle
import time

import numpy as np

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.cluster import GaveUpOnSuffixSearch
from orthogonal_dfa.l_star.examples.spliceai_oracles import (
    CompositionResidualOracle,
    PerLengthResidualOracle,
    SetDifferenceOracle,
    balanced_oracle,
    canonical_oracle,
    load_fm,
)
from orthogonal_dfa.l_star.lstar import (
    counterexample_driven_synthesis,
    denoise_accept_labels,
)
from orthogonal_dfa.l_star.prefix_suffix_tracker import (
    PrefixSuffixTracker,
    SearchConfig,
)
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.statistics import (
    compute_suffix_size_counterexample_gen,
    population_size_and_evidence_margin,
)
from orthogonal_dfa.spliceai.load_model import load_spliceai


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mss", type=float, default=0.1, help="min signal strength (sets eps)"
    )
    p.add_argument("--sampler-len", type=int, default=95, help="prefix/suffix length")
    p.add_argument("--acc-threshold", type=float, default=0.9)
    p.add_argument(
        "--fnr-limit",
        type=float,
        default=0.05,
        help="0.02 is too strict for a real (non-Bernoulli) oracle; use ~0.05",
    )
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--num-prefixes", type=int, default=500)
    p.add_argument("--addtl", type=int, default=300)
    p.add_argument("--model-size", type=int, default=400)
    p.add_argument("--model-seed", type=int, default=0)
    p.add_argument("--eval-count", type=int, default=20000)
    p.add_argument(
        "--n-max", type=int, default=4, help="max k-mer order for residual oracles"
    )
    p.add_argument(
        "--recalibrate",
        action="store_true",
        help="re-center the accept threshold on the median at --sampler-len",
    )
    p.add_argument(
        "--residual",
        action="store_true",
        help="composition-residual oracle (single fit length)",
    )
    p.add_argument(
        "--residual-perlen",
        action="store_true",
        help="length-robust generic-BoW composition residual (per-length bins)",
    )
    p.add_argument(
        "--setdiff",
        choices=["plain", "resid"],
        default=None,
        help="set-difference vs the fixed-motif model (default SpliceAI \\ FM)",
    )
    p.add_argument(
        "--reverse", action="store_true", help="with --setdiff, use FM \\ SpliceAI"
    )
    p.add_argument(
        "--save-dir",
        default=None,
        help="where to write per-round pickles (default runs/<config>)",
    )
    return p.parse_args()


def build_oracle(args, exon, model):
    if args.setdiff:
        fm = load_fm(1)
        ref, nm = args.sampler_len, args.n_max
        if args.setdiff == "resid":
            a = CompositionResidualOracle(exon, model, n_max=nm, ref_len=ref)
            b = CompositionResidualOracle(exon, fm, n_max=nm, ref_len=ref)
        else:
            a = balanced_oracle(model, exon, ref)
            b = balanced_oracle(fm, exon, ref)
        if args.reverse:
            a, b = b, a
        target = "FM \\ SpliceAI" if args.reverse else "SpliceAI \\ FM"
        print(f"oracle: SETDIFF {args.setdiff} {target}", flush=True)
        return SetDifferenceOracle(a, b, exon)
    if args.residual_perlen:
        o = PerLengthResidualOracle(
            exon, model, n_max=args.n_max, len_lo=90, len_hi=2 * args.sampler_len + 5
        )
        print(
            f"oracle: PER-LENGTH RESIDUAL (generic BoW n<={args.n_max}); "
            f"mean per-bin composition R^2={o.composition_r2:.3f}",
            flush=True,
        )
        return o
    if args.residual:
        o = CompositionResidualOracle(
            exon, model, n_max=args.n_max, ref_len=args.sampler_len
        )
        print(
            f"oracle: RESIDUAL (n<={args.n_max}); composition R^2={o.composition_r2:.3f}",
            flush=True,
        )
        return o
    o = (
        balanced_oracle(model, exon, args.sampler_len)
        if args.recalibrate
        else canonical_oracle(model, exon)
    )
    print(f"oracle: SpliceAI (recalibrated={args.recalibrate})", flush=True)
    return o


def default_save_dir(args):
    tag = f"mss{args.mss}_np{args.num_prefixes}_len{args.sampler_len}"
    if args.residual:
        tag += f"_resid{args.n_max}"
    if args.residual_perlen:
        tag += f"_residPL{args.n_max}"
    if args.setdiff:
        tag += f"_setdiff-{args.setdiff}" + ("-rev" if args.reverse else "")
    return os.path.join("runs", tag)


def save_state(save_dir, tag, dfa, dt, pst, args):
    tbl = pst.table
    # tbl._suffixes / tbl._masks are private (no public whole-matrix getter).
    suffixes = [list(s) for s in tbl._suffixes]
    masks = (
        np.array(tbl._masks, dtype=np.int8)
        if tbl._masks
        else np.zeros((0, tbl.num_prefixes), dtype=np.int8)
    )
    state = dict(
        tag=tag,
        prefixes=[list(p) for p in tbl.prefixes],
        representative=np.asarray(tbl.representative),
        suffixes=suffixes,  # the suffix classifiers live in dt's TriPredicate.vs
        masks=masks,  # [n_suffix, n_prefix] int8, -1 = unobserved
        fully_observed=np.asarray(tbl.fully_observed()),
        decision_boundary=pst.decision_boundary,
        evidence_margin=pst.evidence_margin,
        config=pst.config,
        dt=dt,
        dfa=dfa,
        args=vars(args),
    )
    path = os.path.join(save_dir, f"round_{tag}.pkl")
    try:
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(
            f"    saved -> {path}  ({len(suffixes)} suffixes, mask {masks.shape})",
            flush=True,
        )
    except Exception as e:  # never let a pickle failure kill a long run
        print(f"    WARNING: save failed: {e!r}", flush=True)


def reachable_from(dfa, start):
    seen, frontier = set(), [start]
    while frontier:
        s = frontier.pop()
        if s in seen:
            continue
        seen.add(s)
        frontier.extend(dfa.transitions[s].values())
    return seen


def main():
    args = parse_args()
    print("ARGS:", vars(args), flush=True)
    save_dir = args.save_dir or default_save_dir(args)
    os.makedirs(save_dir, exist_ok=True)

    exon = default_exon
    model = load_spliceai(args.model_size, args.model_seed)
    oracle = build_oracle(args, exon, model)

    n, eps = population_size_and_evidence_margin(
        signal_strength=args.mss, acceptable_fpr=0.01, acceptable_fnr=0.01
    )
    config = SearchConfig(
        suffix_family_size=n,
        evidence_margin=eps,
        decision_rule_fpr=0.01,
        suffix_size_counterexample_gen=compute_suffix_size_counterexample_gen(
            0.01, 0.5 + args.mss
        ),
        min_signal_strength=args.mss,
        num_addtl_prefixes=args.addtl,
        fnr_limit=args.fnr_limit,
    )
    print(
        f"config: suffix_family_size={n} eps={eps:.4f}  save_dir={save_dir}", flush=True
    )

    pst = PrefixSuffixTracker.create(
        UniformSampler(args.sampler_len),
        np.random.default_rng(0),
        oracle,
        config,
        num_prefixes=args.num_prefixes,
    )

    # Fixed held-out set for REAL DFA-vs-oracle agreement (not the internal estimate).
    eval_rng = np.random.default_rng(999)
    eval_strings = eval_rng.integers(0, 4, size=(8000, args.sampler_len)).tolist()
    eval_truth = oracle.membership_queries(eval_strings)
    print(
        f"held-out: {len(eval_strings)} @ len {args.sampler_len}, "
        f"oracle accept={eval_truth.mean():.3f}",
        flush=True,
    )

    def report(dfa, dt):
        reach = reachable_from(dfa, dfa.initial_state)
        pred = np.array([dfa.accepts_input(s) for s in eval_strings], dtype=bool)
        print(
            f"    reachable {len(reach)}/{dt.num_states} "
            f"(accepting&reachable {len(reach & set(dfa.final_states))}) | "
            f"REAL agreement={(eval_truth == pred).mean():.4f} "
            f"dfa_accept={pred.mean():.3f}",
            flush=True,
        )

    t0 = time.time()
    last_dfa = last_dt = None
    gen = counterexample_driven_synthesis(
        pst, additional_counterexamples=args.addtl, acc_threshold=args.acc_threshold
    )
    try:
        for i, (dfa, dt, pst_copy) in enumerate(gen):
            last_dfa, last_dt = dfa, dt
            print(
                f"=== round {i}: states={dt.num_states} finals={len(dfa.final_states)} "
                f"elapsed={time.time() - t0:.0f}s ===",
                flush=True,
            )
            report(dfa, dt)
            save_state(save_dir, f"{i:02d}", dfa, dt, pst, args)
            if pst_copy is None:
                print("synthesis terminated on its own", flush=True)
                break
            if i + 1 >= args.max_rounds:
                print(f"hit max-rounds={args.max_rounds}", flush=True)
                break
    except GaveUpOnSuffixSearch as e:
        print(f"\n!!! GaveUpOnSuffixSearch: {e}", flush=True)
        if last_dfa is None:
            raise SystemExit(0)

    dfa = denoise_accept_labels(pst, last_dfa)
    print("\n===== FINAL DFA =====\n", dfa, flush=True)
    save_state(save_dir, "final", dfa, last_dt, pst, args)
    print(f"total time {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
