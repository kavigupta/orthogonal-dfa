"""The two objectives, plus the statistics the internal one is read off of.

``J_internal`` scores how close ``m`` is to being a deterministic automaton:

    max_delta E_{x,c} sum_s m(x)[s] m(xc)[delta(s, c)]
      = E_c sum_s max_{s'} E_x[ m(x)[s] m(xc)[s'] ]

since ``delta(s, c)`` is chosen independently per ``(s, c)``. It is bounded by 1, with
equality exactly when ``m`` is one-hot and consistent with some ``delta``, so hedging
between two copies of a state costs quadratically. It prefers *coarse* partitions and is
maximised by total collapse.

``J_external`` is the log-likelihood of the noisy labels under a per-state accept rate.
It prefers partitions that *refine* accept/reject.

The coarsest partition that is both a transition congruence and refines acceptance is
the Nerode congruence, so the minimal DFA sits at a joint optimum. Any deterministic
refinement of it is also optimal, which is why extraction minifies.
"""

import torch


class TransitionStatistics:
    """EMA of ``T[c, s, s'] = E_x[m(x)[s] m(xc)[s']]`` and support ``N[c, s]``.

    ``delta`` is read off the EMA rather than the current batch: in a cell holding
    little probability mass, a per-batch ``argmax`` over ``s'`` is an argmax over noise,
    and a confidently wrong ``delta`` produces a confidently wrong gradient.
    """

    def __init__(self, alphabet_size, num_states, decay=0.95, device="cpu"):
        self.decay = decay
        self.t = torch.zeros(alphabet_size, num_states, num_states, device=device)
        self.n = torch.zeros(alphabet_size, num_states, device=device)
        self.steps = 0

    def update(self, t_batch, n_batch):
        self.t.mul_(self.decay).add_(t_batch.detach(), alpha=1 - self.decay)
        self.n.mul_(self.decay).add_(n_batch.detach(), alpha=1 - self.decay)
        self.steps += 1

    def support(self):
        return self.n / (1 - self.decay**self.steps)

    def conditional(self):
        """``P(s' | c, s)``, left uniform where there is no support."""
        n = self.support()
        out = (self.t / (1 - self.decay**self.steps)) / n.unsqueeze(-1).clamp_min(1e-12)
        out[n < 1e-12] = 1.0 / out.shape[-1]
        return out

    def transitions(self):
        """``delta[s][c]``, as the transition dict shape ``automata-lib`` wants."""
        delta = self.conditional().argmax(-1).cpu().numpy()
        num_symbols, num_states = delta.shape
        return {
            s: {c: int(delta[c, s]) for c in range(num_symbols)}
            for s in range(num_states)
        }


def batch_transition_statistics(log_m, x, alphabet_size):
    """Per-batch ``T[c, s, s']`` and ``N[c, s]``, each normalised within a symbol.

    The normalisation makes ``sum_{s, s'} T[c, s, s'] == 1``, so ``J_internal`` lands in
    ``[0, 1]`` and a cell's contribution is automatically bounded by its support.
    """
    m = log_m.exp()
    pair = m[:, :-1].unsqueeze(-1) * m[:, 1:].unsqueeze(-2)
    symbols = torch.arange(alphabet_size, device=x.device)
    onehot = (x.unsqueeze(-1) == symbols).to(m.dtype)
    counts = onehot.sum((0, 1)).clamp_min(1e-12)
    t = torch.einsum("blc,blij->cij", onehot, pair) / counts[:, None, None]
    n = torch.einsum("blc,bls->cs", onehot, m[:, :-1]) / counts[:, None]
    return t, n


def internal_objective(t_batch, stats, *, temperature, min_support):
    """``E_c sum_s max_{s'} T[c, s, s']``, maximiser taken from the EMA.

    ``temperature`` softens the max: 0 is the exact argmax, 1 weights by
    ``P(s' | c, s)``. Annealing it down avoids locking onto an early bad argmax. The
    weights are detached, so the gradient hits only the batch's own ``T`` — this is the
    M-step of hard EM.

    Cells below ``min_support`` are dropped rather than contributing a noisy gradient;
    their true contribution is at most their support, so the objective barely moves.
    """
    p = stats.conditional()
    if temperature <= 0:
        weights = torch.zeros_like(p).scatter_(-1, p.argmax(-1, keepdim=True), 1.0)
    else:
        weights = torch.softmax(p.clamp_min(1e-12).log() / temperature, dim=-1)
    keep = (stats.support() >= min_support).to(t_batch.dtype)
    return ((weights * t_batch).sum(-1) * keep).sum(-1).mean()


def internal_information(t_batch):
    """``I(m(xc) ; m(x), c)`` -- determinism that is also *informative*.

    ``internal_objective`` (``E_c sum_s max_s' T``) is maximised by total collapse, which
    is trivially deterministic. That makes merging two states a cheap descent direction:
    it buys the same determinism as refining but in one step, and once ``m`` is one-hot on
    a single state the softmax saturates and ``J_external`` can never undo it.

    Worse, it gives refinement no *incremental* reward. On ``.*1010101.*`` the residual
    nondeterminism is ``R --c--> {R, A}``; every partial split of ``R`` still leaves
    nondeterminism, so the objective sits on a plateau (measured: 0.9853, flat over eight
    rounds) until the whole 7-state chain appears at once.

    Mutual information fixes both. It is 0 at collapse, so merging stops being a descent
    direction at all; and a deterministic partition with ``k`` states scores ``log k``, so
    every additional distinction that stays predictable is rewarded on its own. It prefers
    *over*-refinement, which is harmless: any deterministic refinement of Nerode minifies
    back to the minimal DFA.
    """
    num_symbols = t_batch.shape[0]
    joint = t_batch / num_symbols  # P(c, s, s'), uniform over c
    next_state = joint.sum((0, 1))
    source = joint.sum(-1)
    entropy_next = -(next_state * next_state.clamp_min(1e-12).log()).sum()
    conditional = -(
        joint * (joint.clamp_min(1e-12) / source.unsqueeze(-1).clamp_min(1e-12)).log()
    ).sum()
    return entropy_next - conditional


def external_objective(probs, labels, mask, *, active_lags=None):
    """Log-likelihood of ``label(x[:t + k])`` under ``probs[b, t, k]``, over all ``t, k``.

    ``probs`` comes from :meth:`NeuralDFA.continuation_accept_probs`, i.e. it is conditioned
    on the *observed* continuation ``x[t:t + k]`` rather than marginalised over
    continuations. That is what gives this term any gradient toward refining the acceptance
    partition at all -- see the docstring of :class:`NeuralDFA`.

    None of this costs oracle queries: the target for ``(t, k)`` is the label at position
    ``t + k`` of the same string, which was already bought when the prefix was labelled. The
    observation table's suffix columns are, in effect, already paid for; the earlier version
    of this term simply averaged them away.

    ``active_lags`` restricts training to continuations up to that length. It has to grow:
    the ``k = 0`` column is the only one whose target does not also require learning ``phi``,
    and with all columns live from the start the head reaches "predict 0.5 everywhere" before
    ``m`` learns anything -- which zeroes ``m``'s gradient (``dp/dm[s] = sigma(<u_s, phi>)``
    becomes identical across ``s``) and deadlocks. Growing it mirrors L*, which finds short
    distinguishers before long ones.
    """
    if active_lags is not None:
        probs = probs[:, :, : active_lags + 1]
    num_lags = probs.shape[2] - 1
    # (B, L + 1, K + 1) windows: entry [b, t, k] is the label/weight at position t + k.
    # Zero-padding gives weight 0 wherever t + k runs past the end of the string.
    pad = torch.zeros(
        labels.shape[0], num_lags, device=labels.device, dtype=labels.dtype
    )
    targets = torch.cat([labels, pad], dim=1).unfold(1, num_lags + 1, 1)
    weights = torch.cat([mask, pad], dim=1).unfold(1, num_lags + 1, 1)

    p = probs.clamp(1e-6, 1 - 1e-6)
    log_lik = targets * p.log() + (1 - targets) * (-p).log1p()
    return (log_lik * weights).sum() / weights.sum().clamp_min(1.0)


def confidence_penalty(log_m):
    """``E_x[H(m(x))] - H(E_x[m(x)])``, i.e. ``-I(prefix; state)``.

    ``J_external`` is otherwise satisfied by a *soft mixture*: it reaches the noise floor
    by interpolating accept rates across the simplex without ever committing to discrete
    states, and when ``J_internal`` then demands determinism, collapsing to one state is a
    cheaper way to get there than finding the congruence. So the assignment has to be hard
    before ``J_internal`` turns on.

    Minimising the conditional entropy alone does not achieve that -- one constant state is
    the cheapest hard assignment, and it collapses in the first round. The marginal term is
    the half that forbids it: together they ask for confident assignments that still spread
    over several states.
    """
    flat = log_m.reshape(-1, log_m.shape[-1])
    m = flat.exp()
    conditional = -(m * flat).sum(-1).mean()
    marginal = m.mean(0)
    return conditional + (marginal * marginal.clamp_min(1e-12).log()).sum()


def balance_penalty(log_m):
    """``||E[m m^T] - I/S||_F^2``.

    Zero only for a hard assignment that uses every state equally. Collapse makes the
    Gram matrix rank one, and collapse is otherwise a strong early attractor because
    ``J_internal = 1`` is reachable immediately while ``J_external`` has to be earned.
    Annealed to zero — it is a scaffold, not part of the objective.
    """
    m = log_m.exp().reshape(-1, log_m.shape[-1])
    gram = m.T @ m / m.shape[0]
    target = torch.eye(gram.shape[0], device=gram.device) / gram.shape[0]
    return ((gram - target) ** 2).sum()
