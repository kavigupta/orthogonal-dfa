"""Counterexample-driven training of an amortised state predictor.

The loop keeps the shape of :func:`orthogonal_dfa.l_star.lstar.counterexample_driven_synthesis`
but two of its costs disappear. Sifting is a forward pass instead of one oracle batch per
decision-tree level, and locating a divergence needs no binary search: ``m`` gives the
state at every prefix at once, so a linear scan finds *every* divergence position. Mined
strings feed ``J_internal``, which is unsupervised, so mining costs no oracle queries at
all — the whole query budget goes to labelling prefixes for ``J_external``.

Status. ``parity`` is recovered exactly (accuracy 1.0, 2 states, 88k distinct queries).
``.*1010101.*`` and modulo-9 are not: ``m`` learns the acceptance partition and stops.

Not for want of oracle information -- the same encoder without the state bottleneck reaches
1.0 prefix accuracy on the same labels. The obstacle is that under the uniform length-40
sampling distribution the Nerode refinement is worth almost nothing in likelihood. With a
per-state scalar accept rate ``J_external`` is *exactly* invariant under every refinement of
the acceptance partition, so it contributes zero gradient toward refining it. Conditioning
on the observed continuation (see :class:`NeuralDFA`) removes that exact invariance, but
only ~0.01 nats of it: the model already predicts within 0.01 of the noise floor without
knowing the sub-states, because the continuations that separate them occur with probability
~2^-k. Meanwhile collapsing buys ``J_internal`` 0.02-0.07. The refinement signal loses to
the collapse signal by roughly an order of magnitude either way.

That is the structural gap with L*, which does not optimise a distribution-matched average
loss at all: it constructs *targeted* distinguishing experiments, so a state distinction is
worth a whole counterexample rather than 2^-k of the likelihood.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from orthogonal_dfa.l_star.neural.extract import denoise_accept_labels, extract_dfa
from orthogonal_dfa.l_star.neural.model import NeuralDFA
from orthogonal_dfa.l_star.neural.objective import (
    TransitionStatistics,
    balance_penalty,
    batch_transition_statistics,
    confidence_penalty,
    external_objective,
    internal_information,
    internal_objective,
)
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.structures import Oracle


@dataclass
class NeuralConfig:
    num_states: int = 32
    hidden_size: int = 128
    num_layers: int = 1
    length: int = 40
    num_strings: int = 3000
    mined_cap: int = 3000
    batch_size: int = 64
    lr: float = 3e-3
    rounds: int = 10
    epochs_per_round: int = 8
    # Must dominate J_internal. J_internal only spans [0, 1] and collapse buys it very
    # little, but its trivial basin is broad enough that at equal weight the optimizer
    # slides into collapse anyway -- observed even where the value arithmetic says collapse
    # is a net loss. J_internal is a tie-breaker toward determinism, not the main signal.
    lambda_external: float = 5.0
    # J_internal is satisfiable immediately -- by total collapse, but also by any trivial
    # congruence ("length mod 2", "last symbol") that says nothing about acceptance. So
    # it stays off while J_external establishes what the states have to explain, then
    # ramps in to make that partition deterministic (which can only refine it).
    internal_warmup_rounds: int = 3
    internal_ramp_rounds: int = 3
    # Continuation lengths the accept head conditions on. Needs to reach the length of the
    # shortest distinguishing suffix -- 7 for `.*1010101.*`.
    num_lags: int = 8
    suffix_dim: int = 16
    # False: the max form with num_lags=0 is the most stable configuration measured
    # (J_external held at the noise floor for 12 rounds, no collapse). The information form
    # does prevent collapse as designed, but it then settles on a *language-independent*
    # deterministic congruence instead -- exactly log 2 on `.*1010101.*`, ~log 3 on modulo,
    # with J_external back at base rate. Neither form recovers the Nerode partition; see
    # the module docstring.
    use_information_objective: bool = False
    # Off by default: -I(prefix; state) makes m discrete but is unsupervised, so it wins
    # over the label signal and encodes the wrong partition. The lagged targets are the
    # supervised way to get the same discreteness. Kept as a knob.
    gamma_confidence: float = 0.0
    # Off by default. It was insurance against collapse, but the warmup already handles
    # that, and with num_states over-provisioned "use every state equally" can only be
    # satisfied by duplicating states -- which caps J_internal below 1 and makes the
    # state-level agreement rate meaningless. Kept as a knob for diagnosing collapse.
    beta_balance: float = 0.0
    balance_rounds: int = 3  # rounds over which beta anneals to zero
    temperature_start: float = 1.0
    # Not annealed to 0. A hard argmax in J_internal drives collapse: on `.*1010101.*` a
    # converged run survived tau=0.46 and 0.31 and collapsed the round tau hit 0.21. The
    # softened max keeps J_internal an expectation over P(s'|c,s), which is gentler; the
    # EMA argmax is what extraction reads anyway, so nothing needs tau=0.
    temperature_end: float = 0.3
    ema_decay: float = 0.95
    min_support: float = 1e-3
    sift_strings: int = 20000
    agreement_target: float = 0.9995
    denoise_samples: int = 200
    # Mined counterexamples get labelled and fed to J_external, not just to J_internal.
    # Feeding them only to J_internal (the original design here) aimed the targeting at the
    # term that was never the binding constraint -- and is the one that prefers collapse.
    # This is what makes a mined string a targeted experiment rather than another sample,
    # and the query cost is exactly what buys that.
    # 0 by default: measured, it does not fix the languages that fail (subseq still
    # extracts one state) and it *breaks* the one that works (parity drops to chance,
    # because 10x weight on a skewed mined sample distorts J_external). It also doubles the
    # query count. Kept as a knob -- but see the module docstring: the mining criterion is
    # self-referential, so a mined string is not yet an L*-style counterexample at all.
    label_mined_per_round: int = 0
    mined_boost: float = 10.0
    # 0: measured, ANY positive value breaks parity (1.0 -> 0.5003 at 0.5, 1.0 and 4.0).
    # The flag is only ~48% precise -- the DFA errs on 18.7% of prefixes and noise flips
    # 20%, so P(flagged) = 0.187*0.8 + 0.813*0.2 = 0.313 of which 0.15 are genuine. Half
    # noise would be survivable if it averaged out, but the noise is fixed per string, so
    # upweighting a flagged prefix just re-amplifies the single coin flip that
    # inverse-multiplicity weighting exists to suppress. Kept as a knob; the noise-robust
    # version has to compare populations, not prefixes.
    error_boost: float = 0.0
    seed: int = 0


@dataclass
class Example:
    string: np.ndarray
    labels: np.ndarray
    mask: np.ndarray
    # Multiplier on this example's J_external weight. Mined counterexamples get more than
    # 1: a counterexample is meant to force a split, and at distribution-matched weight it
    # cannot, because the refinement it argues for is worth ~0.01 nats of average
    # likelihood. 0 marks an example with no labels at all.
    boost: float = 1.0
    # 1 at prefixes the current hypothesis DFA gets wrong, 0 elsewhere. This is the
    # oracle-grounded counterexample signal: unlike an m-vs-delta divergence it is evidence
    # about the *language*, not about m's self-consistency.
    error: np.ndarray = None


@dataclass
class LabelCache:
    """Deduplicating label store; ``len`` is the distinct-query count.

    Short prefixes are shared across most sampled strings, so deduplication is what
    keeps the query count near-linear in the number of strings rather than in the number
    of (string, position) pairs.
    """

    oracle: Oracle
    values: dict = field(default_factory=dict)
    total_queries: int = 0

    def prefetch(self, strings):
        missing = sorted({tuple(s) for s in strings} - self.values.keys())
        if not missing:
            return
        answers = self.oracle.membership_queries([list(s) for s in missing])
        self.total_queries += len(missing)
        self.values.update(zip(missing, (bool(a) for a in answers)))

    def __getitem__(self, string):
        return self.values[tuple(string)]

    def __len__(self):
        return len(self.values)


def labelled_examples(strings, labels: LabelCache, boost=1.0) -> List[Example]:
    """Label every prefix, weighting each by inverse multiplicity in the pool.

    The noise is fixed per string, so a prefix has exactly *one* label no matter how
    often it is drawn. Short prefixes are shared by many strings, so unit weighting
    repeats one coin flip hundreds of times and ``J_external`` learns it as ground truth
    — the empty prefix is the extreme case, a population of size one. Weighting by
    ``1 / multiplicity`` makes each distinct oracle answer count once, which is what its
    information content actually is.
    """
    prefixes = [[tuple(s[:t]) for t in range(len(s) + 1)] for s in strings]
    labels.prefetch([list(p) for row in prefixes for p in row])
    out = []
    for s, row in zip(strings, prefixes):
        y = np.array([labels[p] for p in row], dtype=np.float32)
        out.append(
            Example(
                np.asarray(s),
                y,
                np.ones(len(s) + 1, dtype=np.float32),
                boost,
                np.zeros(len(s) + 1, dtype=np.float32),
            )
        )
    return out


def mark_hypothesis_errors(dfa, pool):
    """Flag every prefix where the extracted DFA disagrees with the label already held.

    One simulation per string gives the DFA's verdict at all of its prefixes, so this costs
    no oracle queries at all -- the labels were bought when the prefix was labelled.

    The flag is contaminated: at 20% noise a DFA scoring 0.813 disagrees on ~31% of prefixes
    where a perfect one would disagree on ~20%. But the spurious flags are spread uniformly
    while the real errors are concentrated on exactly the prefixes the hypothesis cannot
    represent, so upweighting them points the gradient at the distinctions a
    distribution-matched loss undervalues.
    """
    for example in pool:
        if not example.boost:
            continue
        state = dfa.initial_state
        for t in range(len(example.string) + 1):
            example.error[t] = float(
                (state in dfa.final_states) != bool(example.labels[t])
            )
            if t < len(example.string):
                state = dfa.transitions[state][int(example.string[t])]


def refresh_weights(pool, error_boost=0.0):
    """Set every labelled example's weight to ``boost / multiplicity`` across the pool.

    Multiplicity has to be counted over the *whole* pool, not per batch: the noise is fixed
    per string, so a prefix has one label however often it is drawn, and short prefixes are
    shared by hundreds of strings. Counting them once each is what stops J_external from
    learning a single coin flip as ground truth.
    """
    counter = Counter()
    for e in pool:
        if e.boost:
            counter.update(tuple(e.string[:t]) for t in range(len(e.string) + 1))
    for e in pool:
        if not e.boost:
            continue
        for t in range(len(e.string) + 1):
            scale = 1.0 + error_boost * float(e.error[t])
            e.mask[t] = e.boost * scale / counter[tuple(e.string[:t])]


def unlabelled_examples(strings) -> List[Example]:
    """Mined strings kept unlabelled -- they drive ``J_internal`` only, and cost nothing."""
    out = []
    for s in strings:
        blank = np.zeros(len(s) + 1, dtype=np.float32)
        out.append(Example(np.asarray(s), blank, blank.copy(), 0.0, blank.copy()))
    return out


def _run_epoch(model, opt, pool, stats, cfg, *, schedule, rng, device):
    order = rng.permutation(len(pool))
    totals, batches = np.zeros(4), 0
    for i in range(0, len(order), cfg.batch_size):
        batch = [pool[j] for j in order[i : i + cfg.batch_size]]
        x = torch.as_tensor(
            np.stack([e.string for e in batch]), dtype=torch.long, device=device
        )
        y = torch.as_tensor(np.stack([e.labels for e in batch]), device=device)
        mask = torch.as_tensor(np.stack([e.mask for e in batch]), device=device)

        log_m = model.state_log_probs(x)
        t_batch, n_batch = batch_transition_statistics(log_m, x, model.alphabet_size)
        # Update before scoring so the first batch has a usable EMA to read delta from.
        stats.update(t_batch, n_batch)

        if cfg.use_information_objective:
            j_internal = internal_information(t_batch)
        else:
            j_internal = internal_objective(
                t_batch,
                stats,
                temperature=schedule["temperature"],
                min_support=cfg.min_support,
            )
        j_external = external_objective(
            model.continuation_accept_probs(
                x, log_m, suffix_grad_scale=schedule["suffix_grad_scale"]
            ),
            y,
            mask,
            active_lags=schedule["active_lags"],
        )
        entropy = confidence_penalty(log_m)
        balance = balance_penalty(log_m)
        loss = (
            -schedule["weight_internal"] * j_internal
            - cfg.lambda_external * j_external
            + schedule["gamma"] * entropy
            + schedule["beta"] * balance
        )

        opt.zero_grad()
        loss.backward()
        opt.step()
        totals += [j_internal.item(), j_external.item(), entropy.item(), balance.item()]
        batches += 1
    return totals / max(batches, 1)


def _simulate(delta, x, initial):
    """State trajectory of the transition table over ``x``, starting at ``initial``."""
    out = torch.empty(x.shape[0], x.shape[1] + 1, dtype=torch.long, device=x.device)
    out[:, 0] = initial
    for t in range(x.shape[1]):
        out[:, t + 1] = delta[x[:, t], out[:, t]]
    return out


def choose_initial_state(model, delta, strings, device):
    """Pick the initial state by self-consistency, not from ``argmax m(empty)``.

    ``m(empty)`` is one decision backed by one oracle label that cannot be averaged, so
    reading the initial state off it inverts the entire trajectory whenever that single
    label was noise-flipped. Every other state is backed by many prefixes. Trying each
    candidate and keeping the one whose simulation best reproduces ``m``'s own
    trajectories decides it from the whole sample instead, and costs no oracle queries.
    """
    with torch.no_grad():
        x = torch.as_tensor(strings, device=device)
        predicted = model.state_log_probs(x).argmax(-1)
        hits = [
            int((predicted == _simulate(delta, x, s)).sum())
            for s in range(model.num_states)
        ]
    return int(np.argmax(hits))


def mine_divergences(
    model, delta, initial, *, cfg, rng, device, alphabet_size, boundary
):
    """``(diverging strings, acceptance agreement, state agreement)``.

    Compares ``m``'s own argmax trajectory against simulating the transition table read
    off ``m``, on fresh uniform strings. Both sides are model objects, so this is free of
    oracle queries — but for the same reason a divergence is evidence of *internal*
    inconsistency only, which is what :func:`denoise_accept_labels` is there to cover.

    Agreement is scored on *acceptance*, matching what
    :func:`orthogonal_dfa.l_star.lstar.estimate_agreement_rate` measures. Raw state
    indices are the wrong unit: a model holding two interchangeable copies of a state can
    be a perfectly good automaton up to minification while its index sequence wanders.
    Mining still uses the state-level divergence, which is the sharper training signal.

    Simulation runs against the pre-minify table so both sides index the same states.
    """
    strings = rng.integers(
        0, alphabet_size, size=(cfg.sift_strings, cfg.length), dtype=np.int64
    )
    diverged = []
    accept_hits, state_hits, positions = 0, 0, 0
    with torch.no_grad():
        accepting = model.accept_probs() > boundary
        for i in range(0, len(strings), 1024):
            chunk = strings[i : i + 1024]
            x = torch.as_tensor(chunk, device=device)
            predicted = model.state_log_probs(x).argmax(-1)
            simulated = _simulate(delta, x, initial)
            same_state = predicted == simulated
            accept_hits += int((accepting[predicted] == accepting[simulated]).sum())
            state_hits += int(same_state.sum())
            positions += same_state.numel()
            diverged.extend(chunk[(~same_state).any(1).cpu().numpy()].tolist())
    return diverged, accept_hits / positions, state_hits / positions


def train_neural_dfa(oracle, cfg: NeuralConfig, *, log=print):
    """Learn a DFA from a noisy membership oracle. Returns ``(dfa, info)``."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NeuralDFA(
        oracle.alphabet_size,
        cfg.num_states,
        cfg.hidden_size,
        cfg.num_layers,
        num_lags=cfg.num_lags,
        suffix_dim=cfg.suffix_dim,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    stats = TransitionStatistics(
        oracle.alphabet_size, cfg.num_states, cfg.ema_decay, device
    )
    labels = LabelCache(oracle)

    sampler = UniformSampler(cfg.length)
    natural = labelled_examples(
        [sampler.sample(rng, oracle.alphabet_size) for _ in range(cfg.num_strings)],
        labels,
    )
    # Mined examples are extra and bounded: they skew hard toward the tail, and training
    # on them alone round after round makes the state assignment non-stationary.
    mined = deque(maxlen=cfg.mined_cap)
    counterexamples: List[Example] = []
    log(f"labelled {len(labels):,} distinct prefixes from {len(natural):,} strings")

    dfa, boundary, agreement = None, 0.5, 0.0
    for round_idx in range(cfg.rounds):
        since_warmup = round_idx - cfg.internal_warmup_rounds
        weight_internal = min(
            1.0, max(0.0, (since_warmup + 1) / cfg.internal_ramp_rounds)
        )
        # Hold the max soft until J_internal is actually on, or the argmax freezes onto
        # whatever partition the warmup happened to leave behind.
        progress = min(
            1.0,
            max(
                0.0, since_warmup / max(cfg.rounds - cfg.internal_warmup_rounds - 1, 1)
            ),
        )
        temperature = (
            cfg.temperature_start
            * (cfg.temperature_end / cfg.temperature_start) ** progress
        )
        schedule = {
            "weight_internal": weight_internal,
            "temperature": temperature,
            "beta": cfg.beta_balance * max(0.0, 1 - round_idx / cfg.balance_rounds),
            # Full strength by the end of the warmup, then held: J_internal wants a hard
            # assignment too, so there is no reason to relax it afterwards.
            # One more continuation length per round: k=0 alone first, so the accept head
            # cannot flatten before m has learned anything.
            "active_lags": min(cfg.num_lags, round_idx),
            # Held at 0 for the first round a continuation length is live, then ramped.
            "suffix_grad_scale": min(1.0, max(0.0, (round_idx - 1) / 3)),
            "gamma": cfg.gamma_confidence
            * min(1.0, (round_idx + 1) / max(cfg.internal_warmup_rounds, 1)),
        }

        pool = natural + counterexamples + list(mined)
        if dfa is not None and cfg.error_boost:
            mark_hypothesis_errors(dfa, pool)
        refresh_weights(pool, cfg.error_boost)
        scores = np.zeros(4)
        for _ in range(cfg.epochs_per_round):
            scores = _run_epoch(
                model,
                opt,
                pool,
                stats,
                cfg,
                schedule=schedule,
                rng=rng,
                device=device,
            )

        delta = stats.conditional().argmax(-1)
        probe = rng.integers(
            0, oracle.alphabet_size, size=(1024, cfg.length), dtype=np.int64
        )
        initial = choose_initial_state(model, delta, probe, device)
        dfa, boundary = extract_dfa(model, stats, initial=initial)
        diverged, agreement, state_agreement = mine_divergences(
            model,
            delta,
            initial,
            cfg=cfg,
            rng=rng,
            device=device,
            alphabet_size=oracle.alphabet_size,
            boundary=boundary,
        )
        log(
            f"round {round_idx}: J_int={scores[0]:.4f} J_ext={scores[1]:.4f} "
            f"negMI={scores[2]:.4f} gram={scores[3]:.4f} "
            f"w_int={weight_internal:.2f} lags={schedule['active_lags']} "
            f"agree={agreement:.5f} state_agree={state_agreement:.5f} "
            f"states={len(dfa.states)} mined={len(diverged)}"
        )
        # Agreement is only meaningful once J_internal is fully on; during warmup a
        # degenerate m can agree with its own degenerate transition table.
        if weight_internal >= 1.0 and agreement >= cfg.agreement_target:
            break
        # The labelled slice is the targeted experiment; the rest is free consistency data.
        if cfg.label_mined_per_round:
            counterexamples.extend(
                labelled_examples(
                    diverged[: cfg.label_mined_per_round], labels, cfg.mined_boost
                )
            )
        mined.extend(
            unlabelled_examples(diverged[cfg.label_mined_per_round :][: cfg.mined_cap])
        )

    before = len(dfa.states), len(dfa.final_states)
    dfa = denoise_accept_labels(
        dfa,
        oracle,
        rng,
        cfg.length,
        boundary=boundary,
        max_samples=cfg.denoise_samples,
    )
    info = {
        "distinct_queries": len(labels),
        "total_queries": labels.total_queries,
        "agreement": agreement,
        "states": len(dfa.states),
        "states_before_denoise": before[0],
        "boundary": boundary,
        "denoise_changed_labels": before != (len(dfa.states), len(dfa.final_states)),
    }
    return dfa, info
