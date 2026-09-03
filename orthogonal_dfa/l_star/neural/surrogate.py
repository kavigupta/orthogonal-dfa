"""Surrogate-guided DFA learning: fit the observation table, then read the DFA off it.

The object L* spends its queries on is the table ``M[i][j] = oracle(p_i + v_j)`` -- 75% of
them directly, and most of the rest classifying fresh prefixes against suffix families,
which is the same thing for rows not yet in the table. The exploitable structure is that
the noiseless table depends on ``p_i`` only through its Nerode state, so it has at most
``n`` distinct rows out of however many prefixes there are.

So: fit a model of the table from a subset of cells, and let it fill the rest.

    M[p][v] = sum_s m(p)[s] * sigma(<u_s, phi(v)>)

``m(p)`` is a distribution over ``S`` row-clusters, ``phi(v)`` embeds a suffix, and ``u_s``
is a state's response profile. The softmax on ``m`` is the "at most n distinct rows"
constraint and is what denoises: per-string noise is not expressible through an ``S``-cluster
bottleneck, so a cell's prediction pools evidence from every prefix in its cluster. That
matters because the noise here is fixed per string -- re-querying a cell can never denoise
it, only populations can.

Rows and columns are *encoders* over the strings rather than free per-row embeddings. That
is the difference from ordinary matrix completion: the model generalises to prefixes and
suffixes it never queried, which is what makes a fresh prefix classifiable for free.

Empirically (``.*1010101.*``, 2000x64 table, symmetric noise 0.8): at 50% of cells the
surrogate predicts held-out cells at 0.973 against the *noiseless* truth, versus 0.8 for
reading each cell's own noisy label, and recovers 80% of the Nerode state information.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import scipy.stats
import torch
from automata.fa.dfa import DFA
from torch import nn

from orthogonal_dfa.l_star.neural.extract import accept_threshold
from orthogonal_dfa.l_star.structures import Oracle


@dataclass
class SurrogateConfig:
    num_states: int = 32  # over-provisioned on purpose; see module notes
    signal_strength: float = 0.3  # same input SearchConfig takes
    suffix_dim: int = 16
    hidden_size: int = 128
    prefix_length: int = 40
    num_prefixes: int = 1200
    num_suffixes: int = 40
    max_suffix_length: int = 10
    # Set by a power calculation, not by feel. Splitting stalls when a block gets small: at
    # k blocks over P prefixes each side of a candidate column holds density * P/(2k) cells,
    # so z = 0.6 * sqrt(density * P / k) for the 0.2-vs-0.8 gap. At density 0.3 with P=1500
    # and k=9 that is 4.2, just under the Bonferroni threshold -- which is why every variant
    # stalled at 4-7 blocks no matter what the test read. At 0.6 it is ~6.0.
    initial_density: float = 0.6
    cells_per_round: int = 6000
    new_suffixes_per_round: int = 6
    # Row growth, OFF by default. The idea is right -- random prefixes find a rare state only
    # in proportion to its rarity -- but this implementation is not: it takes a row where the
    # DFA disagrees with the surrogate, and the DFA is *extracted from* the surrogate, so a
    # divergence measures extraction error rather than language error. Early on almost every
    # probe "diverges", so the rows are near-arbitrary and each gets only cols_per_new_prefix
    # columns, diluting the pool with under-observed rows. Measured over 14 targets it moved
    # >=0.97 from 4/14 to 3/14 and median accuracy 0.855 -> 0.818 at more queries, with
    # modulo9 falling 1.0 -> 0.773. A grounded version has to buy the candidate row's cells
    # and keep it only if the OBSERVED row contradicts the hypothesis.
    probe_strings: int = 3000
    new_prefixes_per_round: int = 0
    # Enrichment: add prefixes landing in under-populated clusters, and buy extra columns
    # for prefixes the model cannot confidently place. Both are statements about our own
    # sample rather than about the language, so neither can be self-referentially wrong the
    # way a DFA-vs-surrogate divergence is.
    enrich_per_round: int = 400
    enrich_candidates: int = 20000
    underconfident_rows_per_round: int = 300
    # Mean absolute difference below which two prototype rows count as the same Nerode state.
    # Rows converge to the noise rates, so the gap between "same" and "different" is
    # |p_1 - p_0|; half of that is a natural cut.
    merge_tolerance: float = 0.15
    # Conformal calibration: fraction of observed cells held out of training and used to
    # bound the surrogate's rate error, and the miscoverage level of that bound.
    # Half the cells. The split test reads only held-out cells (testing on the cells the
    # proposal was fitted to is circular), so this directly sets its power: at 0.15 the test
    # under-splits badly -- modulo9 found 3 blocks of 9, subseq 2 of 8.
    calibration_fraction: float = 0.5
    # Split down from one blob (a state exists only if some suffix proved it) versus merging
    # up from the softmax's clusters. Splitting is the better-founded design -- merging does
    # 496 pairwise tests corrected only across columns, so chance differences become
    # permanent states -- but it measures WORSE end to end: subseq 0.813 on every seed
    # against 1.0 on one seed for merging. Off until that is understood.
    use_split_from_blob: bool = False
    split_test_on_holdout: bool = True
    cross_fit_folds: int = 4
    conformal_alpha: float = 0.1
    # SearchConfig uses 0.001, but it tests over far larger samples. Here a (cluster, suffix)
    # cell holds ~15 observations, so a real 0.2-vs-0.8 gap gives z ~ 3.3 while Bonferroni
    # over 40 columns at 0.001 demands 4.2 -- genuine differences get missed and clusters
    # collapse. Measured over 3 seeds: subseq reaches exactly 8 states on 2/3 seeds at 0.01
    # versus 1/3 at 0.001. Loosening further hurts (0/3 at 0.05), because extra groups make
    # the transition argmax noisier and the DFA minifies back down.
    split_pvalue: float = 0.01
    # Statistical transition resolution, OFF by default: implemented but NOT validated.
    # Enabling it regressed parity from 1.0 to 0.5003. The motivation stands -- delta(b, c)
    # needs the group of p . c, which is not a row of the table, so the argmax infers it
    # entirely from the model -- but buying successor cells per transition is not yet
    # producing better transitions than the co-occurrence vote.
    resolve_transitions_statistically: bool = False
    successors_per_transition: int = 25
    # Columns bought per successor batch. The decision is a MAX over columns, so restricting
    # it to a subset directly removes chances to catch the one suffix that distinguishes.
    # 0 means every suffix.
    transition_columns: int = 0
    min_cells_per_estimate: int = 8
    refine_iters: int = 12
    # The correction should cover the tests actually run, not a hypothetical 60 blocks:
    # 40 columns x 60 blocks put the critical z at 4.6 and blocked splits that were real.
    max_blocks: int = 16
    cols_per_new_prefix: int = 16
    rounds: int = 5
    steps_per_round: int = 250
    prefix_batch: int = 256
    lr: float = 3e-3
    seed: int = 0


class Encoder(nn.Module):
    """Sequence -> vector, with a learned readout for the empty sequence.

    The empty string has to be a first-class input: it is the initial state's row and the
    plain-acceptance column.
    """

    def __init__(self, alphabet_size, out_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(alphabet_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.h0 = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.head = nn.Linear(hidden_size, out_dim)

    def forward(self, padded, lengths):
        return self.all_positions(padded)[
            torch.arange(len(lengths), device=padded.device), lengths
        ]

    def all_positions(self, padded):
        """``(B, L + 1, out)`` -- every prefix of every row in one pass.

        Counterexample search needs the verdict at all L+1 prefixes of a probe string; one
        causal pass gives them instead of L+1 separate encodes.
        """
        h0 = self.h0.expand(-1, padded.shape[0], -1).contiguous()
        out, _ = self.gru(self.embed(padded), h0)
        return self.head(torch.cat([h0[-1].unsqueeze(1), out], dim=1))


class TableSurrogate(nn.Module):
    """``M[p][v] = sum_s m(p)[s] * sigma(<u_s, phi(v)>)``.

    A mixture over states, not a mixture of logits: a prefix *has* a state and a state *has*
    a behaviour. Mixing logits would let a prefix interpolate between states and mean nothing.
    """

    def __init__(self, alphabet_size, cfg: SurrogateConfig):
        super().__init__()
        self.rows = Encoder(alphabet_size, cfg.num_states, cfg.hidden_size)
        self.cols = Encoder(alphabet_size, cfg.suffix_dim, cfg.hidden_size)
        self.state_vectors = nn.Parameter(torch.randn(cfg.num_states, cfg.suffix_dim))
        # Response confined to [0.5 - s, 0.5 + s], the only rates a cell can legitimately
        # have. Without this the model memorises: predicted rates pile up at 0.0 and 1.0
        # instead of the noise rates, which only beats predicting the rate if the model is
        # right about individual cells -- and it gets there by grouping prefixes whose
        # observed cells happen to agree, which shreds the partition (each true residue
        # smeared over 16-21 clusters, transition concentration 0.25).
        #
        # The bound must be FIXED, not learned. A learned range is vacuous: the model just
        # widens it to 0/1 and memorises anyway, which is exactly what happened.
        # min_signal_strength is an input to the existing pipeline too (SearchConfig), so
        # this is the same information L* is given.
        self.signal_strength = cfg.signal_strength

    def row_log_probs(self, padded, lengths):
        return torch.log_softmax(self.rows(padded, lengths), dim=-1)

    def response(self, col_padded, col_lengths):
        """``(V, S)``: probability that state ``s`` followed by suffix ``v`` is accepted.

        Confined to the achievable rate range; see ``signal_strength`` above.
        """
        inner = torch.sigmoid(self.cols(col_padded, col_lengths) @ self.state_vectors.T)
        return 0.5 - self.signal_strength + 2 * self.signal_strength * inner

    def forward(self, row_log_m, response, rows_idx, cols_idx):
        return (row_log_m[rows_idx].exp() * response[cols_idx]).sum(-1)


def pad(sequences, width, device="cpu"):
    out = np.zeros((len(sequences), width), dtype=np.int64)
    for i, s in enumerate(sequences):
        out[i, : len(s)] = s
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    return torch.as_tensor(out, device=device), torch.as_tensor(lengths, device=device)


@dataclass
class CellPool:
    """The prefix set, suffix set, and which cells have actually been paid for."""

    oracle: Oracle
    prefixes: List[List[int]]
    suffixes: List[List[int]]
    answers: dict = field(default_factory=dict)  # (i, j) -> bool
    queried: set = field(default_factory=set)  # distinct strings, for the query count
    holdout: set = field(
        default_factory=set
    )  # cells withheld from training, for conformal

    def observe(self, cells):
        wanted = [(i, j) for i, j in cells if (i, j) not in self.answers]
        if not wanted:
            return 0
        strings = [self.prefixes[i] + self.suffixes[j] for i, j in wanted]
        results = self.oracle.membership_queries(strings)
        for (i, j), string, value in zip(wanted, strings, results):
            self.answers[(i, j)] = bool(value)
            self.queried.add(tuple(string))
        return len(wanted)

    def training_arrays(self):
        """Observed cells minus the conformal holdout -- calibration must be on unseen data."""
        usable = sorted(set(self.answers) - self.holdout)
        cells = np.array(usable, dtype=np.int64)
        labels = np.array([self.answers[tuple(c)] for c in cells], dtype=np.float32)
        return cells, labels


def _fit(model, opt, pool, *, cfg, rng, device, steps):
    cells, labels = pool.training_arrays()
    rows = torch.as_tensor(cells[:, 0], device=device)
    cols = torch.as_tensor(cells[:, 1], device=device)
    targets = torch.as_tensor(labels, device=device)
    p_pad, p_len = pad(pool.prefixes, cfg.prefix_length, device)
    v_pad, v_len = pad(pool.suffixes, cfg.max_suffix_length, device)

    for _ in range(steps):
        # Minibatch over PREFIXES, not cells: encoding the prefix set dominates the cost,
        # and every cell in the batch reuses the same encodings.
        chosen = torch.as_tensor(
            rng.choice(
                len(pool.prefixes),
                size=min(cfg.prefix_batch, len(pool.prefixes)),
                replace=False,
            ),
            device=device,
        )
        keep = torch.isin(rows, chosen)
        if not bool(keep.any()):
            continue
        remap = torch.full((len(pool.prefixes),), -1, dtype=torch.long, device=device)
        remap[chosen] = torch.arange(len(chosen), device=device)

        row_log_m = model.row_log_probs(p_pad[chosen], p_len[chosen])
        response = model.response(v_pad, v_len)
        predicted = model(row_log_m, response, remap[rows[keep]], cols[keep]).clamp(
            1e-6, 1 - 1e-6
        )
        y = targets[keep]
        loss = -(y * predicted.log() + (1 - y) * (1 - predicted).log()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def _encode_all(model, pool, cfg, device, alphabet_size):
    """Row distributions for every prefix and every one-symbol extension of it."""
    with torch.no_grad():
        p_pad, p_len = pad(pool.prefixes, cfg.prefix_length, device)
        current = model.row_log_probs(p_pad, p_len).exp()
        successors = []
        for symbol in range(alphabet_size):
            extended = [p + [symbol] for p in pool.prefixes]
            e_pad, e_len = pad(extended, cfg.prefix_length + 1, device)
            successors.append(model.row_log_probs(e_pad, e_len).exp())
    return current, torch.stack(successors)


def refine_partition(prefix_rows, successor_rows, empty_col, boundary, cfg):
    """Moore/Hopcroft refinement over surrogate-predicted rows.

    Start from the acceptance partition and split a block only when its transition is
    ill-defined -- i.e. its members' ``c``-successors fall in different blocks. Every split is
    then justified by an actual successor disagreement, so the result is a congruence *by
    construction*.

    That is the fix for letting the softmax choose the partition outright. The cell loss
    cannot distinguish a Nerode class from that class split three ways -- all the sub-clusters
    learn the same prototype row and predict identical cells -- so descent splits it at random,
    and the successors then scatter across the sub-clusters (measured: 44/44 (state, symbol)
    pairs diffuse, top-1 share 0.15-0.4). Refinement never creates a split it cannot justify.

    The surrogate supplies the expensive ingredient: the predicted row of ``p . c`` for
    prefixes that were never queried, so "see where (s, c) lands" costs nothing.
    """
    block = (prefix_rows[:, empty_col] > boundary).astype(np.int64)
    successors = None
    for _ in range(cfg.refine_iters):
        num_blocks = int(block.max()) + 1
        centroids = np.stack(
            [
                (
                    prefix_rows[block == b].mean(0)
                    if (block == b).any()
                    else np.zeros(prefix_rows.shape[1])
                )
                for b in range(num_blocks)
            ]
        )
        # Classify each p.c into a block by nearest centroid -- the surrogate standing in for
        # sifting a fresh prefix through the discrimination tree.
        successors = np.stack(
            [
                np.abs(rows[:, None, :] - centroids[None]).mean(-1).argmin(1)
                for rows in successor_rows
            ]
        )
        signature = list(zip(block, *successors))
        codes = {sig: i for i, sig in enumerate(sorted(set(signature)))}
        if len(codes) > cfg.max_blocks:
            break
        refined = np.array([codes[sig] for sig in signature], dtype=np.int64)
        if len(codes) == num_blocks:
            break
        block = refined
    return block, successors


def conformal_rate_bound(response, clusters, pool, holdout, cfg):
    """Distribution-free bound on how far the surrogate's predicted RATE can be off.

    Conformal on the *label* is hopeless here: at 20% per-string noise a fifth of cells carry
    a nonconformity score near 1, so any band with 90% label coverage is the whole interval
    and every cell comes back undecided. The noise is irreducible per cell.

    The rate is the right target -- it is what the cluster bottleneck estimates and what
    pooling makes accurate. Held-out cells grouped by ``(cluster, suffix)`` give an empirical
    rate to score against, and the ``1 - alpha`` quantile of ``|predicted - empirical|`` is a
    calibrated bound. Unlike the two thresholds I picked by hand (a mean row distance, and
    "any successor disagreement"), this one is measured.
    """
    buckets = {}
    for i, j in holdout:
        buckets.setdefault((int(clusters[i]), j), []).append(pool.answers[(i, j)])
    # The empirical rate is itself a Bernoulli estimate, so |predicted - empirical| includes
    # the calibration set's own sampling error -- at n=8 that alone is ~0.14 std, which would
    # put q_hat near half the 0.6 gap between the noise rates and make everything undecided.
    # Subtract that off so the score measures model error rather than our own noise.
    scores = []
    for (state, suffix), ys in buckets.items():
        if len(ys) < cfg.min_cells_per_estimate:
            continue
        empirical = float(np.mean(ys))
        standard_error = np.sqrt(max(empirical * (1 - empirical), 1e-6) / len(ys))
        scores.append(
            max(0.0, abs(response[state, suffix] - empirical) - 1.96 * standard_error)
        )
    if len(scores) < 20:
        return 1.0  # not enough calibration data to claim anything
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - cfg.conformal_alpha)) / n)
    return float(np.quantile(scores, level))


def resolve_transitions(pool, block_of, num_groups, accepts, totals, *, cfg, rng):
    """Determine delta(b, c) from cells bought for the successors themselves.

    The argmax over the co-occurrence tensor cannot work: delta(b, c) needs the group of
    ``p . c``, but ``p . c`` is generally not a row of the table, so its group comes entirely
    from the model. L* sidesteps this by keeping the table prefix-closed, so the successor is
    a real row. Bounded version of the same thing: there are only ``num_groups * |Sigma|``
    transitions, so buy cells for a sample of successors per transition and assign the
    destination by the same two-proportion test used to form the groups.

    Falls back to the group's own index (a self-loop) when nothing clears the test, which
    keeps the DFA total without inventing a destination.
    """
    members = {b: np.flatnonzero(block_of == b) for b in range(num_groups)}
    delta = np.arange(num_groups)[None, :].repeat(pool.oracle.alphabet_size, axis=0)
    for b, rows in members.items():
        # Only prefixes with room to grow: p . c must still fit the padded width.
        rows = np.array([i for i in rows if len(pool.prefixes[i]) < cfg.prefix_length])
        if rows.size == 0:
            continue
        sample = rng.choice(
            rows, size=min(cfg.successors_per_transition, len(rows)), replace=False
        )
        for c in range(pool.oracle.alphabet_size):
            successors = [pool.prefixes[i] + [c] for i in sample]
            start = len(pool.prefixes)
            pool.prefixes.extend(successors)
            width = cfg.transition_columns or len(pool.suffixes)
            columns = rng.choice(
                len(pool.suffixes), size=min(width, len(pool.suffixes)), replace=False
            )
            pool.observe(
                [(start + k, int(j)) for k in range(len(successors)) for j in columns]
            )
            # Pool the successors into one profile and ask which group it fails to differ from.
            succ_accepts = np.zeros(len(pool.suffixes))
            succ_totals = np.zeros(len(pool.suffixes))
            for k in range(len(successors)):
                for j in columns:
                    value = pool.answers.get((start + k, int(j)))
                    if value is not None:
                        succ_accepts[j] += value
                        succ_totals[j] += 1
            stacked_a = np.vstack([accepts, succ_accepts])
            stacked_t = np.vstack([totals, succ_totals])
            # BEST match, not "any group we fail to reject". Failing to reject is not
            # evidence of sameness: with only cols_per_new_prefix columns observed the test
            # has little power, so the successor fails to differ from most groups and an
            # arbitrary tie-break decides -- previously the largest group, which funnelled
            # every transition into one state and left the rest unreachable (parity 0.5003).
            scored = [
                (
                    difference_evidence(
                        stacked_a,
                        stacked_t,
                        num_groups,
                        b2,
                        min_cells=cfg.min_cells_per_estimate,
                    )[0],
                    b2,
                )
                for b2 in range(num_groups)
                if len(members[b2])
            ]
            scored = [(z, b2) for z, b2 in scored if np.isfinite(z)]
            if scored:
                delta[c, b] = min(scored)[1]
    return delta


def trim_unreachable(dfa):
    """Drop states unreachable from the initial state. Deliberately NOT ``minify``.

    ``minify`` also merges behaviourally equivalent states, which assumes the transition
    table is correct. Here it is estimated, so two states whose transitions are wrong in the
    same way look equivalent and get merged -- turning a transition error into total
    collapse, which is what the recurring "1 state" outcome was. It also explains why
    loosening ``split_pvalue`` made things worse: more groups meant noisier transitions,
    hence more spurious equivalences for minify to collapse.

    ``transition_resolver._to_dfa_and_tree`` does not minify either. It does not need to: its
    states are discrimination-tree leaves, each separated by an actual distinguishing suffix,
    so no two are equivalent by construction. ``merge_by_counts`` gives the same guarantee.
    """
    reachable, frontier = {dfa.initial_state}, [dfa.initial_state]
    while frontier:
        state = frontier.pop()
        for target in dfa.transitions[state].values():
            if target not in reachable:
                reachable.add(target)
                frontier.append(target)
    renamed = {state: i for i, state in enumerate(sorted(reachable))}
    return DFA(
        states=set(renamed.values()),
        input_symbols=set(dfa.input_symbols),
        transitions={
            renamed[s]: {c: renamed[t] for c, t in dfa.transitions[s].items()}
            for s in reachable
        },
        initial_state=renamed[dfa.initial_state],
        final_states={renamed[s] for s in dfa.final_states if s in reachable},
        allow_partial=False,
    )


def observed_counts(pool, clusters, num_states):
    """``(accepts, totals)`` per ``(cluster, suffix)`` over the cells actually queried."""
    accepts = np.zeros((num_states, len(pool.suffixes)))
    totals = np.zeros((num_states, len(pool.suffixes)))
    for (i, j), value in pool.answers.items():
        # Rows appended after the assignment was computed (successor rows bought by
        # resolve_transitions) have no cluster yet; they are counted next round.
        if i >= len(clusters):
            continue
        accepts[clusters[i], j] += value
        totals[clusters[i], j] += 1
    return accepts, totals


def difference_evidence(accepts, totals, s, t, *, min_cells):
    """Largest z-statistic over the columns both ``s`` and ``t`` have enough cells for.

    ``-inf`` when no column qualifies. Returning the statistic rather than a verdict is what
    lets a caller pick the *best-matching* group instead of any group it merely fails to
    reject -- failing to reject is not evidence of sameness, and at low power everything
    fails to reject.
    """
    usable = (totals[s] >= min_cells) & (totals[t] >= min_cells)
    if not usable.any():
        return -np.inf, 0
    rate_s = accepts[s][usable] / totals[s][usable]
    rate_t = accepts[t][usable] / totals[t][usable]
    pooled = (accepts[s][usable] + accepts[t][usable]) / (
        totals[s][usable] + totals[t][usable]
    )
    standard_error = np.sqrt(
        np.maximum(pooled * (1 - pooled), 1e-9)
        * (1 / totals[s][usable] + 1 / totals[t][usable])
    )
    return float((np.abs(rate_s - rate_t) / standard_error).max()), int(usable.sum())


def differ_significantly(accepts, totals, s, t, *, pvalue, min_cells):
    """Do clusters ``s`` and ``t`` differ on ANY suffix, by a two-proportion test?

    Max over columns, not mean. That distinction is the whole point: separating two states
    typically needs one specific suffix out of forty -- for ``.*1010101.*``, the one that
    completes the pattern from one state and not the other. Averaging over columns buries it,
    which is why a mean row distance over-merged and why the cross-entropy itself barely
    notices the distinction (measured: each column separates ~12 of 28 state pairs, but the
    other ~37 columns agree and dominate the loss).

    Tested on the cells actually observed rather than on the model's predicted rows, so a
    decision never rests on a value the surrogate interpolated. Bonferroni over the columns
    compared.
    """
    z, columns = difference_evidence(accepts, totals, s, t, min_cells=min_cells)
    if not columns:
        return False
    return bool(z > scipy.stats.norm.isf(pvalue / (2 * columns)))


def cross_fitted_rows(pool, cfg, rng, device, alphabet_size):
    """Out-of-fold predicted rows: each cell predicted by a model that never saw it.

    A single train/test split forces a trade-off I created and then measured: the split test
    must read cells the proposal was not fitted to, but every cell withheld is a cell the
    proposer loses. At 15% holdout the model proposed well (subseq blocks 9-10, accuracy up
    to 1.0) and at 50% it proposed badly (blocks 4-6, accuracy 0.813) -- while the test itself
    made no difference, since testing on all cells under-split identically.

    K folds remove the trade-off: every model trains on (K-1)/K of the cells, and every cell
    is predicted by the one model that excluded it.
    """
    cells = sorted(pool.answers)
    fold_of = {
        cell: i % cfg.cross_fit_folds
        for i, cell in enumerate(rng.permutation(len(cells)))
    }
    rows = np.zeros((len(pool.prefixes), len(pool.suffixes)))
    for k in range(cfg.cross_fit_folds):
        held = {cells[i] for i, f in fold_of.items() if f == k}
        pool.holdout = held
        model = TableSurrogate(alphabet_size, cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        _fit(
            model, opt, pool, cfg=cfg, rng=rng, device=device, steps=cfg.steps_per_round
        )
        with torch.no_grad():
            p_pad, p_len = pad(pool.prefixes, cfg.prefix_length, device)
            v_pad, v_len = pad(pool.suffixes, cfg.max_suffix_length, device)
            predicted = (
                (
                    model.row_log_probs(p_pad, p_len).exp()
                    @ model.response(v_pad, v_len).T
                )
                .cpu()
                .numpy()
            )
        for i, j in held:
            rows[i, j] = predicted[i, j]
    return rows


def split_by_evidence(pool, prefix_rows, boundary, cfg):
    """Start with every prefix in one block; split only where the cells prove a difference.

    Merging *down* from the softmax's clusters cannot avoid over-splitting: with 32 clusters
    that is 496 pairwise comparisons, and any pair that differs by chance stays apart. Here
    every block that exists exists because some suffix separated it, so a state is never
    created without evidence -- which is what the discrimination tree gives L*.

    The surrogate proposes and the cells decide. For a candidate suffix, the surrogate's
    predicted rate says which side of that column each prefix falls on; the two sides are then
    compared on the cells actually observed at that column. The split happens only if that
    two-proportion test clears a Bonferroni threshold over every (column, split) tried, so the
    surrogate can suggest a bad column without being able to create a state.

    Returns a block label per prefix.
    """
    # HELD-OUT cells only. The split direction comes from a model fitted to the training
    # cells, so testing on those same cells is circular: a prefix whose cell was noise-flipped
    # gets predicted onto the other side of the column, and the test then reads that same
    # flipped cell and confirms the split. That manufactures states out of noise, which is
    # exactly what splitting-from-one-blob is supposed to make impossible.
    accepts = np.zeros((len(pool.prefixes), len(pool.suffixes)))
    totals = np.zeros((len(pool.prefixes), len(pool.suffixes)))
    for i, j in pool.holdout:
        if i < len(prefix_rows):
            accepts[i, j] = pool.answers[(i, j)]
            totals[i, j] = 1

    block = np.zeros(len(prefix_rows), dtype=np.int64)
    queue = [0]
    comparisons = max(len(pool.suffixes) * cfg.max_blocks, 1)
    critical = scipy.stats.norm.isf(cfg.split_pvalue / (2 * comparisons))

    while queue and block.max() + 1 < cfg.max_blocks:
        current = queue.pop(0)
        rows = np.flatnonzero(block == current)
        if len(rows) < 2 * cfg.min_cells_per_estimate:
            continue
        best = (critical, None, None)
        for v in range(len(pool.suffixes)):
            side = prefix_rows[rows, v] > boundary
            if side.all() or not side.any():
                continue
            counts_hi = totals[rows[side], v].sum()
            counts_lo = totals[rows[~side], v].sum()
            if min(counts_hi, counts_lo) < cfg.min_cells_per_estimate:
                continue
            rate_hi = accepts[rows[side], v].sum() / counts_hi
            rate_lo = accepts[rows[~side], v].sum() / counts_lo
            pooled = (accepts[rows[side], v].sum() + accepts[rows[~side], v].sum()) / (
                counts_hi + counts_lo
            )
            standard_error = np.sqrt(
                max(pooled * (1 - pooled), 1e-9) * (1 / counts_hi + 1 / counts_lo)
            )
            z = abs(rate_hi - rate_lo) / standard_error
            if z > best[0]:
                best = (z, v, side)
        if best[1] is None:
            continue
        fresh = block.max() + 1
        block[rows[best[2]]] = fresh
        queue.extend([current, fresh])
    return block, int(block.max()) + 1


def merge_by_counts(pool, clusters, counts, cfg):
    """Group clusters unless some suffix separates them significantly on observed cells."""
    accepts, totals = observed_counts(pool, clusters, len(counts))
    order = np.argsort(-counts)
    labels = np.full(len(counts), -1)
    representatives = []
    for state in order:
        if counts[state] == 0:
            continue
        for group, rep in enumerate(representatives):
            if not differ_significantly(
                accepts,
                totals,
                state,
                rep,
                pvalue=cfg.split_pvalue,
                min_cells=cfg.min_cells_per_estimate,
            ):
                labels[state] = group
                break
        else:
            labels[state] = len(representatives)
            representatives.append(state)
    labels[labels < 0] = max(len(representatives), 1)
    return labels, max(len(representatives), 1) + 1


def merge_by_conformal(response, counts, boundary, q_hat):
    """Group clusters unless some suffix separates them *with calibrated confidence*.

    ``label(s, v)`` is 1 when even the pessimistic rate ``R[s][v] - q_hat`` sits above the
    boundary, 0 when the optimistic one sits below, and undecided in between -- the same
    accept/reject/None trichotomy ``TriPredicate`` uses.

    Two clusters merge unless some suffix gives them confident and *differing* labels. That
    fixes the flaw in the mean-distance version: a single distinguishing column now blocks a
    merge outright instead of being averaged away across 40 suffixes.
    """
    confident = np.where(
        response - q_hat > boundary, 1, np.where(response + q_hat < boundary, 0, -1)
    )
    order = np.argsort(-counts)
    labels = np.full(len(counts), -1)
    representatives = []
    for state in order:
        if counts[state] == 0:
            continue
        for group, rep in enumerate(representatives):
            both = (confident[state] >= 0) & (confident[rep] >= 0)
            if not np.any(both & (confident[state] != confident[rep])):
                labels[state] = group
                break
        else:
            labels[state] = len(representatives)
            representatives.append(state)
    labels[labels < 0] = max(len(representatives), 1)
    return labels, max(len(representatives), 1) + 1


def merge_by_row(response, counts, tolerance):
    """Group states whose prototype rows agree, returning a label per state.

    The cell loss is *invariant* to arbitrary refinement: if a Nerode class is split across
    three clusters, all three learn the same prototype row, predict the same cells and incur
    the same loss, so gradient descent splits it arbitrarily. An arbitrary refinement is not
    deterministic -- prefixes in one sub-cluster scatter across the sub-clusters of the next
    class -- which is why the raw successor distribution is diffuse (measured: 44/44 pairs
    non-concentrated, top-1 share 0.15-0.4).

    ``minify`` cannot fix this because it needs correct transitions first. Rows can be
    compared directly: two states with the same response over every suffix are the same
    Nerode state, by definition.
    """
    order = np.argsort(-counts)
    labels = np.full(len(counts), -1)
    representatives = []
    for state in order:
        if counts[state] == 0:
            continue
        for group, rep in enumerate(representatives):
            if np.abs(response[state] - response[rep]).mean() < tolerance:
                labels[state] = group
                break
        else:
            labels[state] = len(representatives)
            representatives.append(state)
    # Unused states get their own sink group so indexing stays total.
    labels[labels < 0] = len(representatives)
    return labels, len(representatives) + 1


def extract_dfa(model, pool, cfg, device, alphabet_size, *, rng=None):
    """Soft-vote the transitions off the softmax clusters, then minify.

    ``T[c, s, t] = sum_p m(p)[s] m(p.c)[t]``, and ``delta(s, c) = argmax_t``. This is a
    population vote, so individual misclassification is tolerated -- but it also *discards*
    the disagreement, which is the signal L*'s ``_resolve`` acts on: prefixes of ``s``
    landing in different places means ``s`` needs splitting.

    Two attempts to use that signal both did worse, in opposite directions, and both for the
    same reason -- neither had a statistical decision rule:

    * ``merge_by_row``: group states whose prototype rows are close. Over-merges, because a
      mean over 40 suffixes dilutes a single distinguishing column. subseq and two_subseq
      collapsed to one state; modulo9 fell to 0.227 on one seed. (It did get parity to
      exactly 2 states, so row comparison does collapse duplicates -- the operator is wrong,
      not the idea.)
    * ``refine_partition``: Moore refinement, splitting a block when its members' successors
      land in different blocks. Over-splits, because the successor assignment is a noisy
      nearest-centroid classification, so any noise-driven disagreement causes a split and
      blocks multiply each round. Parity shattered into 14-21 states at 0.49 accuracy.

    What both are missing is ``transition_resolver._splits``: split only when the
    disagreement is *significant* under a binomial test calibrated to the noise
    (``split_pval``, ``decision_rule_fpr``, ``evidence_margin``). Under per-string noise a
    bare threshold is not enough to decide whether two prefixes differ.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    current, _ = _encode_all(model, pool, cfg, device, alphabet_size)
    with torch.no_grad():
        v_pad, v_len = pad(pool.suffixes, cfg.max_suffix_length, device)
        response = model.response(v_pad, v_len).T.cpu().numpy()  # (S, V)
        empty_col = [j for j, v in enumerate(pool.suffixes) if len(v) == 0][0]

    clusters = current.argmax(-1).cpu().numpy()
    counts = np.bincount(clusters, minlength=cfg.num_states)
    # Hard counts, not soft mass: soft mass is never exactly zero, so unused clusters would
    # enter the 2-means split carrying arbitrary accept probabilities.
    boundary = accept_threshold(response[:, empty_col], counts.astype(float))

    q_hat = conformal_rate_bound(response, clusters, pool, pool.holdout, cfg)
    if cfg.cross_fit_folds > 1:
        saved = pool.holdout
        prefix_rows = cross_fitted_rows(pool, cfg, rng, device, alphabet_size)
        pool.holdout = saved
    else:
        prefix_rows = current.cpu().numpy() @ response
    if cfg.use_split_from_blob:
        block_of, num_groups = split_by_evidence(pool, prefix_rows, boundary, cfg)
    else:
        labels, num_groups = merge_by_counts(pool, clusters, counts, cfg)
        block_of = labels[clusters]

    # A block's profile is the mean predicted row of its members. Assigning p.c to the
    # nearest profile is the sifting step -- classifying a prefix that is not a table row --
    # which is the one thing only the surrogate can do without buying cells.
    profiles = np.stack(
        [
            (
                prefix_rows[block_of == b].mean(0)
                if (block_of == b).any()
                else np.full(prefix_rows.shape[1], 0.5)
            )
            for b in range(num_groups)
        ]
    )
    _, successors = _encode_all(model, pool, cfg, device, alphabet_size)
    delta = np.zeros((alphabet_size, num_groups), dtype=int)
    for c in range(alphabet_size):
        successor_rows = successors[c].cpu().numpy()[: len(block_of)] @ response
        # MAX over columns, not mean -- the same distinction that governs merging. A
        # successor matches its own block on every column (gap ~0.03) and differs sharply
        # from any other on the few distinguishing ones (gap ~0.51, measured). Averaging over
        # 40 columns shrinks that 0.51 to ~0.04, where it loses to noise, so successors get
        # assigned to arbitrary blocks and most blocks end up unreachable.
        landed = np.abs(successor_rows[:, None, :] - profiles[None]).max(-1).argmin(1)
        for b in range(num_groups):
            members = block_of == b
            if members.any():
                delta[c, b] = np.bincount(
                    landed[members], minlength=num_groups
                ).argmax()

    accept = np.array(
        [
            (
                prefix_rows[block_of == b, empty_col].mean()
                if (block_of == b).any()
                else 0.5
            )
            for b in range(num_groups)
        ]
    )
    empty_row = [i for i, p in enumerate(pool.prefixes) if len(p) == 0][0]
    dfa = DFA(
        states=set(range(num_groups)),
        input_symbols=set(range(alphabet_size)),
        transitions={
            s: {c: int(delta[c, s]) for c in range(alphabet_size)}
            for s in range(num_groups)
        },
        initial_state=int(block_of[empty_row]),
        final_states={s for s in range(num_groups) if accept[s] > boundary},
        allow_partial=False,
    )
    return trim_unreachable(dfa), boundary, {"q_hat": q_hat, "groups": num_groups}


def counterexample_prefixes(
    model, pool, dfa, boundary, *, cfg, rng, device, alphabet_size
):
    """Prefixes where the extracted DFA disagrees with the surrogate.

    The surrogate is the denoised classifier here -- it plays the discrimination tree's role
    in L*, and unlike a single oracle answer it is not 20% wrong. A divergence marks a prefix
    the hypothesis routes to the wrong state, so adding it as a row targets exactly the
    under-populated states that random prefix sampling misses. The new rows then get real
    cells bought for them, which is what keeps this grounded rather than self-referential.
    """
    probes = rng.integers(
        0, alphabet_size, size=(cfg.probe_strings, cfg.prefix_length), dtype=np.int64
    )
    with torch.no_grad():
        v_pad, v_len = pad(pool.suffixes, cfg.max_suffix_length, device)
        empty_col = [j for j, v in enumerate(pool.suffixes) if len(v) == 0][0]
        accept = model.response(v_pad, v_len)[empty_col]
        x = torch.as_tensor(probes, device=device)
        rows = torch.log_softmax(model.rows.all_positions(x), dim=-1).exp()
        surrogate_accepts = ((rows * accept).sum(-1) > boundary).cpu().numpy()

    found = []
    for row, verdict in zip(probes, surrogate_accepts):
        state = dfa.initial_state
        for t in range(len(row) + 1):
            if (state in dfa.final_states) != bool(verdict[t]):
                found.append(row[:t].tolist())
                break
            if t < len(row):
                state = dfa.transitions[state][int(row[t])]
    rng.shuffle(found)
    return found[: cfg.new_prefixes_per_round]


def enrich_underpopulated(model, pool, *, cfg, rng, device, alphabet_size):
    """Prefixes that land in clusters with too few members.

    This is ``enrich_underrepresented_leaves`` in the surrogate's terms, and it targets the
    measured failure mode: a cluster with few members can fit its cells' noise instead of
    pooling it away, because moving its prototype row costs nothing elsewhere.

    Screening costs no oracle queries -- the encoder places a candidate prefix on its own --
    so a large candidate pool can be filtered for free and only the keepers get cells bought.
    """
    with torch.no_grad():
        p_pad, p_len = pad(pool.prefixes, cfg.prefix_length, device)
        counts = np.bincount(
            model.row_log_probs(p_pad, p_len).argmax(-1).cpu().numpy(),
            minlength=cfg.num_states,
        )
    occupied = counts[counts > 0]
    if occupied.size == 0:
        return []
    target = float(np.median(occupied))

    candidates = [
        rng.integers(
            0, alphabet_size, size=rng.integers(0, cfg.prefix_length + 1)
        ).tolist()
        for _ in range(cfg.enrich_candidates)
    ]
    with torch.no_grad():
        c_pad, c_len = pad(candidates, cfg.prefix_length, device)
        placed = model.row_log_probs(c_pad, c_len).argmax(-1).cpu().numpy()

    # Rarest-cluster-first, so the thinnest clusters get filled before the merely below-median.
    wanted = [i for i, c in enumerate(placed) if counts[c] < target]
    wanted.sort(key=lambda i: counts[placed[i]])
    return [candidates[i] for i in wanted[: cfg.enrich_per_round]]


def underconfident_rows(model, pool, *, cfg, device):
    """Prefixes whose cluster assignment is least certain, by entropy of ``m(p)``.

    A row the model cannot place is a row whose observed cells do not yet pin it to any
    prototype -- so buying more of its columns is exactly the missing evidence.
    """
    with torch.no_grad():
        p_pad, p_len = pad(pool.prefixes, cfg.prefix_length, device)
        log_m = model.row_log_probs(p_pad, p_len)
        entropy = -(log_m.exp() * log_m).sum(-1).cpu().numpy()
    return np.argsort(-entropy)[: cfg.underconfident_rows_per_round]


def propose_suffixes(model, pool, weights, *, cfg, rng, device, alphabet_size):
    """Candidate suffixes scored by how much they split the current clusters.

    This is the surrogate standing in for L*'s distinguisher search: instead of sampling
    suffix families and evaluating each over every prefix, score candidates by prediction and
    only pay for cells on the winners.
    """
    candidates = [
        rng.integers(
            0, alphabet_size, size=rng.integers(1, cfg.max_suffix_length + 1)
        ).tolist()
        for _ in range(64)
    ]
    known = {tuple(v) for v in pool.suffixes}
    candidates = [v for v in candidates if tuple(v) not in known]
    if not candidates:
        return []
    with torch.no_grad():
        c_pad, c_len = pad(candidates, cfg.max_suffix_length, device)
        response = model.response(c_pad, c_len).cpu().numpy()  # (C, S)
    mass = weights / max(weights.sum(), 1e-9)
    mean = (response * mass).sum(-1, keepdims=True)
    spread = (((response - mean) ** 2) * mass).sum(-1)
    order = np.argsort(-spread)[: cfg.new_suffixes_per_round]
    return [candidates[i] for i in order]


def learn_dfa(oracle: Oracle, cfg: SurrogateConfig, *, log=print) -> Tuple[DFA, dict]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphabet_size = oracle.alphabet_size

    prefixes = [[]] + [
        rng.integers(
            0, alphabet_size, size=rng.integers(0, cfg.prefix_length + 1)
        ).tolist()
        for _ in range(cfg.num_prefixes - 1)
    ]
    suffixes = [[]] + [
        rng.integers(
            0, alphabet_size, size=rng.integers(1, cfg.max_suffix_length + 1)
        ).tolist()
        for _ in range(cfg.num_suffixes - 1)
    ]
    pool = CellPool(oracle, prefixes, suffixes)

    all_cells = [(i, j) for i in range(len(prefixes)) for j in range(len(suffixes))]
    initial = rng.permutation(len(all_cells))[
        : int(cfg.initial_density * len(all_cells))
    ]
    pool.observe([all_cells[k] for k in initial])

    observed = sorted(pool.answers)
    pool.holdout.update(
        tuple(observed[k])
        for k in rng.permutation(len(observed))[
            : int(cfg.calibration_fraction * len(observed))
        ]
    )

    model = TableSurrogate(alphabet_size, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    dfa, boundary = None, 0.5
    for round_idx in range(cfg.rounds):
        _fit(
            model,
            opt,
            pool,
            cfg=cfg,
            rng=rng,
            device=device,
            steps=cfg.steps_per_round,
        )
        dfa, boundary, diag = extract_dfa(
            model, pool, cfg, device, alphabet_size, rng=rng
        )
        log(
            f"round {round_idx}: cells={len(pool.answers):,} queries={len(pool.queried):,} "
            f"prefixes={len(pool.prefixes)} suffixes={len(pool.suffixes)} "
            f"q_hat={diag['q_hat']:.3f} groups={diag['groups']} states={len(dfa.states)}"
        )
        if round_idx == cfg.rounds - 1:
            break

        new_rows = counterexample_prefixes(
            model,
            pool,
            dfa,
            boundary,
            cfg=cfg,
            rng=rng,
            device=device,
            alphabet_size=alphabet_size,
        )
        new_rows = new_rows + enrich_underpopulated(
            model, pool, cfg=cfg, rng=rng, device=device, alphabet_size=alphabet_size
        )
        fresh_cells = []
        for index in underconfident_rows(model, pool, cfg=cfg, device=device):
            for j in rng.choice(
                len(pool.suffixes),
                size=min(cfg.cols_per_new_prefix, len(pool.suffixes)),
                replace=False,
            ):
                fresh_cells.append((int(index), int(j)))
        for prefix in new_rows:
            index = len(pool.prefixes)
            pool.prefixes.append(prefix)
            for j in rng.choice(
                len(pool.suffixes),
                size=min(cfg.cols_per_new_prefix, len(pool.suffixes)),
                replace=False,
            ):
                fresh_cells.append((index, int(j)))
        pool.observe(fresh_cells)

        current, _ = _encode_all(model, pool, cfg, device, alphabet_size)
        for suffix in propose_suffixes(
            model,
            pool,
            current.sum(0).cpu().numpy(),
            cfg=cfg,
            rng=rng,
            device=device,
            alphabet_size=alphabet_size,
        ):
            pool.suffixes.append(suffix)

        # Spend the round's budget where the surrogate is least sure, plus full coverage of
        # any newly added column so it is anchored in real answers rather than prediction.
        with torch.no_grad():
            p_pad, p_len = pad(pool.prefixes, cfg.prefix_length, device)
            v_pad, v_len = pad(pool.suffixes, cfg.max_suffix_length, device)
            predicted = (
                (
                    model.row_log_probs(p_pad, p_len).exp()
                    @ model.response(v_pad, v_len).T
                )
                .cpu()
                .numpy()
            )
        uncertainty = -np.abs(predicted - 0.5)
        for i, j in pool.answers:
            uncertainty[i, j] = -np.inf
        flat = np.argsort(uncertainty.ravel())[::-1][: cfg.cells_per_round]
        pool.observe(
            [(int(k // predicted.shape[1]), int(k % predicted.shape[1])) for k in flat]
        )

    return dfa, {
        "distinct_queries": len(pool.queried),
        "cells": len(pool.answers),
        "prefixes": len(pool.prefixes),
        "suffixes": len(pool.suffixes),
        "states": len(dfa.states),
        "boundary": round(float(boundary), 4),
        "q_hat": round(float(diag["q_hat"]), 4),
        "groups": diag["groups"],
    }
