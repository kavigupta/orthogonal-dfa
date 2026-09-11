import torch
from torch import nn


class NeuralDFA(nn.Module):
    """Amortised state predictor ``m(x)`` over prefixes, plus a per-state accept rate.

    ``m`` is a distribution over ``num_states`` latent states for *every* prefix of the
    input, produced in one pass. This is the difference from the unrolled relaxations in
    ``orthogonal_dfa/utils/pdfa.py`` and ``orthogonal_dfa/deep_dfa``: nothing is
    backpropagated through 40 soft transition matrices, so the transition function is
    fit from a local consistency constraint rather than an end-to-end rollout.

    The accept head is ``sigma(<u_s, phi(v)>)``: the probability that state ``s`` followed
    by continuation ``v`` is accepted. It is unconstrained, so it converges to the *noisy*
    rate (``p_1`` / ``p_0`` under :class:`AsymmetricBernoulli`) and the noise parameters
    never have to be supplied.

    Conditioning on ``v`` is what makes ``J_external`` able to see state structure at all.
    A per-state scalar ``a[s]`` is *exactly invariant* under splitting a state into two
    halves with equal accept rates, so it contributes literally zero gradient toward
    refining the acceptance partition -- and marginalising over ``v`` by continuation
    *length* only (an earlier version of this head) is nearly as invariant, because on
    ``.*1010101.*`` the sub-states of "not yet matched" differ in accept-after-k-random-steps
    by O(2^-7). Indexing by the continuation itself removes the invariance: two states must
    now agree on *specific* ``v``, which is the Myhill-Nerode condition.

    ``phi`` is a compact learned recurrent encoding, not one column per suffix: nothing is
    enumerated, and only the continuations that actually occur in the data are evaluated.
    The empty continuation keeps a free per-state scalar instead of going through ``phi``,
    for the reason noted at ``accept_logits``.
    """

    def __init__(
        self,
        alphabet_size,
        num_states,
        hidden_size=128,
        num_layers=1,
        *,
        num_lags=8,
        suffix_dim=16,
    ):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.num_states = num_states
        self.num_lags = num_lags
        self.embed = nn.Embedding(alphabet_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.h0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size) * 0.02)
        self.head = nn.Linear(hidden_size, num_states)
        # Must not start tied. With every state's accept rate equal, p is constant in m, so
        # the gradient reaching m's logits is the same for every state and the softmax
        # Jacobian kills it exactly -- and while m stays near-uniform every state's rate
        # gets the same gradient too, so the deadlock holds. Random init breaks it.
        # The empty continuation keeps its own free scalar per state rather than being
        # folded into the bilinear head. Routing it through a shared vector lets one
        # parameter flatten every state's accept rate at once, which zeroes m's gradient
        # far faster than flattening S independent scalars does -- measured as a total
        # failure to learn even parity.
        self.accept_logits = nn.Parameter(torch.randn(num_states))
        self.state_vectors = nn.Parameter(torch.randn(num_states, suffix_dim))
        self.suffix_embed = nn.Embedding(alphabet_size, suffix_dim)
        self.suffix_gru = nn.GRU(suffix_dim, suffix_dim, batch_first=True)

    def state_log_probs(self, x):
        """``x``: ``(B, L)`` int64. Returns ``(B, L + 1, S)`` log-probabilities.

        Index ``t`` is the prefix ``x[:t]``, so index 0 is the empty prefix and index
        ``L`` is the whole string.
        """
        h0 = self.h0.expand(-1, x.shape[0], -1).contiguous()
        out, _ = self.gru(self.embed(x), h0)
        out = torch.cat([h0[-1].unsqueeze(1), out], dim=1)
        return torch.log_softmax(self.head(out), dim=-1)

    def forward(self, x):
        return self.state_log_probs(x)

    def empty_prefix_log_probs(self):
        """``(S,)`` log-probabilities for the empty prefix, i.e. the initial state."""
        return torch.log_softmax(self.head(self.h0[-1, 0]), dim=-1)

    def accept_probs(self):
        """``(S,)`` accept probability per state, i.e. the empty continuation.

        This is what extraction reads; the longer continuations exist only to supervise the
        partition.
        """
        return torch.sigmoid(self.accept_logits)

    def continuation_encodings(self, x):
        """``(B, L + 1, K, d)``: encoding of the continuation ``x[t:t + k]``, ``k >= 1``.

        A recurrent encoder, not a bag of symbol embeddings. An additive encoding is
        monotone in the symbol counts, so ``sigma(<u_s, phi(v)>)`` cannot represent
        ``parity(v)`` -- which made every ``k >= 1`` column an unfittable target on the
        parity oracle and drowned out ``k = 0``. A GRU represents modular counting fine.

        Since ``x[t:t + k]`` for increasing ``k`` is a prefix of the window at ``t``, one
        pass over the ``K``-wide sliding windows yields every length at once. Entries where
        ``t + k`` runs past the end of the string are garbage and the loss masks them out.
        """
        batch, length = x.shape
        tail = torch.zeros(batch, self.num_lags, dtype=x.dtype, device=x.device)
        windows = torch.cat([x, tail], dim=1).unfold(1, self.num_lags, 1)
        encoded, _ = self.suffix_gru(
            self.suffix_embed(windows.reshape(-1, self.num_lags))
        )
        return encoded.reshape(batch, length + 1, self.num_lags, -1)

    def continuation_accept_probs(self, x, log_m, *, suffix_grad_scale=1.0):
        """``(B, L + 1, K + 1)``: ``P(accept | prefix x[:t], continuation x[t:t + k])``.

        The state is latent, so this is a genuine mixture over states rather than a mixture
        of logits: ``sum_s m(x[:t])[s] * sigma(<u_s, phi(v)>)``.
        """
        m = log_m.exp()
        empty = (m * torch.sigmoid(self.accept_logits)).sum(-1, keepdim=True)
        if self.num_lags == 0:
            return empty
        # Same value as m, but its gradient is scaled. When a continuation length switches
        # on, u_s and the suffix GRU are still at random init, and their large initial
        # errors flow straight back into m and destroy it -- measured on `.*1010101.*` as a
        # collapse the round lag 2 activated, with J_internal still switched off entirely.
        # Ramping this from 0 lets the head learn to *use* m before it can push back on it.
        damped = suffix_grad_scale * m + (1 - suffix_grad_scale) * m.detach()
        per_state = torch.sigmoid(
            torch.einsum(
                "btkd,sd->btks", self.continuation_encodings(x), self.state_vectors
            )
        )
        return torch.cat(
            [empty, torch.einsum("bts,btks->btk", damped, per_state)], dim=2
        )
