from dataclasses import dataclass

import numpy as np
import pythomata
import torch
from automata.fa.dfa import DFA
from automata.fa.nfa import NFA
from torch import nn

from orthogonal_dfa.utils.probability import ZeroProbability


@dataclass
class InitializedPDFA:
    initial_state_probs: torch.Tensor  # (num_states,)
    transition_probs: torch.Tensor  # (num_states, input_channels, num_states)
    accepting_state_logprobs: torch.Tensor  # (num_states,)


def pdfa_forward(ipdfa: InitializedPDFA, input_probs: torch.Tensor):
    return pdfa_forward_fast(
        ipdfa.initial_state_probs,
        ipdfa.transition_probs,
        ipdfa.accepting_state_logprobs,
        input_probs,
    )


def pdfa_forward_slow(
    initial_state_probs, transition_probs, accepting_state_logprobs, input_probs
):
    N, L, C = input_probs.shape
    [S] = initial_state_probs.shape
    assert transition_probs.shape == (S, C, S)
    assert accepting_state_logprobs.shape == (S,)

    p_state = initial_state_probs.unsqueeze(0).expand(N, -1)  # (N, num_states)
    for i in range(L):
        # b = batch
        # n = next state
        # p = previous state
        # c = input channel
        # p_state[b, n] = sum_p sum_c p_state[b, p] * x[b, i, c] * transition_probs[p, c, n]
        p_state = torch.einsum(
            "bp,bc,pcn->bn", p_state, input_probs[:, i, :], transition_probs
        )
    log_p_state = torch.log(p_state + 1e-20)  # (N, S)
    log_acceptance = torch.logsumexp(
        log_p_state + accepting_state_logprobs.unsqueeze(0), dim=1
    )  # (N,)
    return log_acceptance


def batched_iterated_matrix_multiply(matrices):
    """
    Performs batched iterated matrix multiplication.

    :param matrices: (N, L, M, M) tensor of N batches of L matrices of size MxM.
    :return: (N, M, M) tensor of the result of multiplying the L matrices for each batch.
    """
    N, L, M, _ = matrices.shape
    if L == 1:
        return matrices[:, 0, :, :]
    if L % 2 == 1:
        first = matrices[:, 0, :, :]
        rest = batched_iterated_matrix_multiply(matrices[:, 1:, :, :])
        return torch.bmm(first, rest)
    multiplied = torch.bmm(
        matrices[:, 0::2, :, :].contiguous().view(-1, M, M),
        matrices[:, 1::2, :, :].contiguous().view(-1, M, M),
    ).view(N, L // 2, M, M)
    return batched_iterated_matrix_multiply(multiplied)


def pdfa_forward_fast(
    initial_state_probs, transition_probs, accepting_state_logprobs, input_probs
):
    _, _, C = input_probs.shape
    [S] = initial_state_probs.shape
    assert transition_probs.shape == (S, C, S)
    assert accepting_state_logprobs.shape == (S,)

    transition_probs_each = torch.einsum(
        "pcn,blc->blpn", transition_probs, input_probs
    )  # (N, L, S, S)

    transition_overall = batched_iterated_matrix_multiply(transition_probs_each)
    p_state = torch.matmul(
        transition_overall.permute(0, 2, 1), initial_state_probs
    )  # (N, S)
    log_p_state = torch.log(p_state + 1e-20)  # (N, S)
    log_acceptance = torch.logsumexp(
        log_p_state + accepting_state_logprobs.unsqueeze(0), dim=1
    )  # (N,)
    return log_acceptance


class PDFA(nn.Module):

    @classmethod
    def from_dfa(cls, dfa: pythomata.SimpleDFA, zero_prob: ZeroProbability):
        """
        Creates a PDFA from a given deterministic finite automaton (DFA).

        :param dfa: pythomata.SimpleDFA instance representing the DFA.
        :return: PDFA instance corresponding to the given DFA.
        """
        num_states = len(dfa.states)
        states = sorted(dfa.states)
        alphabet = sorted(dfa.alphabet)
        state_to_index = {state: i for i, state in enumerate(states)}
        input_channels = len(dfa.alphabet)

        logit_initial_state_probs = torch.full(
            (num_states,), zero_prob.logit_probability
        )
        logit_initial_state_probs[state_to_index[dfa.initial_state]] = (
            -zero_prob.logit_probability
        )

        logit_transition_probs = torch.full(
            (num_states, input_channels, num_states), zero_prob.logit_probability
        )
        for state in states:
            state_index = state_to_index[state]
            for symbol in alphabet:
                next_state = dfa.transition_function[state][symbol]
                next_state_index = state_to_index[next_state]
                symbol_index = alphabet.index(symbol)
                logit_transition_probs[state_index, symbol_index, next_state_index] = (
                    -zero_prob.logit_probability
                )

        logit_accepting_state_probs = torch.full(
            (num_states,), zero_prob.logit_probability
        )
        for state in dfa.accepting_states:
            state_index = state_to_index[state]
            logit_accepting_state_probs[state_index] = -zero_prob.logit_probability

        return cls(
            logit_initial_state_probs,
            logit_transition_probs,
            logit_accepting_state_probs,
        )

    @classmethod
    def create(cls, input_channels, num_states):
        """
        Creates a PDFA with randomly initialized parameters.

        :param input_channels: int, the number of input channels.
        :param num_states: int, the number of states in the PDFA.
        :return: PDFA instance with randomly initialized parameters.
        """
        logit_initial_state_probs = torch.randn((num_states,))
        logit_transition_probs = torch.randn((num_states, input_channels, num_states))
        logit_accepting_state_probs = torch.randn((num_states,))
        return cls(
            logit_initial_state_probs,
            logit_transition_probs,
            logit_accepting_state_probs,
        )

    def __init__(
        self,
        logit_initial_state_probs,
        logit_transition_probs,
        logit_accepting_state_probs,
    ):
        """
        A probabilistic DFA with uniform initial state distribution and uniform transition probabilities.
        """
        super().__init__()
        self.logit_initial_state_probs = nn.Parameter(logit_initial_state_probs)
        self.logit_transition_probs = nn.Parameter(logit_transition_probs)
        self.logit_accepting_state_probs = nn.Parameter(logit_accepting_state_probs)

    @property
    def initialized(self) -> InitializedPDFA:
        return InitializedPDFA(
            initial_state_probs=nn.functional.softmax(
                self.logit_initial_state_probs, dim=0
            ),
            transition_probs=nn.functional.softmax(self.logit_transition_probs, dim=2),
            accepting_state_logprobs=torch.nn.LogSigmoid()(
                self.logit_accepting_state_probs
            ),
        )

    def forward(self, log_input_probs):
        """
        Outputs (1, N) tensor of log-probabilities of acceptance for each input sequence.

        The 1 dimension is for forward compatibility if we decide to batch multiple PDFAs.
        """

        return pdfa_forward(self.initialized, torch.exp(log_input_probs))[None]


def hyperbolic_sigmoid(x):
    return x / (1 + torch.abs(x)) * 0.5 + 0.5


def hyperbolic_softmax(logits, dim=0):
    x = hyperbolic_sigmoid(logits)
    x_sum = torch.sum(x, dim=dim, keepdim=True)
    return x / x_sum


class PDFAHyberbolicParameterization(nn.Module):

    @classmethod
    def create(cls, input_channels, num_states):
        return cls(
            torch.randn((num_states,)),
            torch.randn((num_states, input_channels, num_states)),
            torch.randn((num_states,)),
        )

    def __init__(
        self,
        invh_initial_state_probs,
        invh_transition_probs,
        invh_accepting_state_probs,
    ):
        """
        A probabilistic DFA with uniform initial state distribution and uniform transition probabilities.
        """
        super().__init__()
        self.invh_initial_state_probs = nn.Parameter(invh_initial_state_probs)
        self.invh_transition_probs = nn.Parameter(invh_transition_probs)
        self.invh_accepting_state_probs = nn.Parameter(invh_accepting_state_probs)

    @property
    def initialized(self) -> InitializedPDFA:
        return InitializedPDFA(
            initial_state_probs=hyperbolic_softmax(
                self.invh_initial_state_probs, dim=0
            ),
            transition_probs=hyperbolic_softmax(self.invh_transition_probs, dim=2),
            accepting_state_logprobs=torch.log(
                hyperbolic_sigmoid(self.invh_accepting_state_probs)
            ),
        )

    def forward(self, log_input_probs):
        return pdfa_forward(self.initialized, torch.exp(log_input_probs))[None]


def to_dfa_for_viz(ipdfa: InitializedPDFA, noise_floor: float) -> DFA:
    """
    Does not create an actual DFA, instead creates a DFA where certain transitions are labeled with e.g.,
        2@76% to indicate that the transition goes to state 2 with 76% probability.

    :param noise_floor: float, the minimum probability to consider a transition as existing, also the threshold
        for accepting states and 1 - noise_floor for the probability that a transition does not need to be
        annotated at.
    """
    prob_initial = ipdfa.initial_state_probs.detach().cpu().numpy()
    prob_transition = ipdfa.transition_probs.detach().cpu().numpy()
    prob_accepting = torch.exp(ipdfa.accepting_state_logprobs).detach().cpu().numpy()

    def state_label(i):
        result = f"S{i}"
        if prob_initial[i] >= noise_floor:
            result += f"\nInit@{prob_initial[i]*100:.0f}%"
        if prob_accepting[i] >= noise_floor:
            result += f"\nAcc@{prob_accepting[i]*100:.0f}%"
        return result

    states = [state_label(i) for i in range(len(prob_initial))]
    transitions = {}
    for s_from in range(prob_transition.shape[0]):
        for s_to in range(prob_transition.shape[2]):
            for c in range(prob_transition.shape[1]):
                p = prob_transition[s_from, c, s_to]
                if p >= noise_floor:
                    if p < 1 - noise_floor:
                        label = f"{c}@{p*100:.0f}%"
                    else:
                        label = f"{c}"
                    if s_from == 2 and c == 4:
                        print(s_to, p, label)
                    trans_this = transitions.setdefault(states[s_from], {})
                    if label not in trans_this:
                        trans_this[label] = set()
                    trans_this[label].add(states[s_to])
    dfa = NFA(
        states=set(states),
        input_symbols={t for ts in transitions.values() for t in ts},
        transitions=transitions,
        initial_state=states[np.argmax(prob_initial)],
        final_states=set(
            s for i, s in enumerate(states) if prob_accepting[i] >= noise_floor
        ),
    )
    return dfa


def entropy_of_logits(logits, dim=0):
    """
    Computes the entropy of a categorical distribution defined by the given logits.

    :param logits: tensor of logits defining the categorical distribution.
    :param dim: int, the dimension along which to compute the entropy.
    :return: tensor of entropies.
    """
    probs = nn.functional.softmax(logits, dim=dim)
    log_probs = nn.functional.log_softmax(logits, dim=dim)
    entropy = -torch.sum(probs * log_probs, dim=dim)
    return entropy


class PDFAOutputtingProbabilities(nn.Module):

    def __init__(self, pdfa_module: PDFA):
        """
        A wrapper around PDFA that outputs probabilities instead of log-probabilities.
        """
        super().__init__()
        self.pdfa = pdfa_module

    def forward(self, log_input_probs):
        log_probs = self.pdfa(log_input_probs)
        return torch.exp(log_probs)
