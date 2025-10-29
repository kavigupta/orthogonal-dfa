import pythomata
import torch
from torch import nn

from orthogonal_dfa.utils.probability import ZeroProbability


def pdfa(
    logit_initial_state_probs: torch.Tensor,
    logit_transition_probs: torch.Tensor,
    logit_accepting_state_probs: torch.Tensor,
    log_input_probs: torch.Tensor,
):
    """
    Computes the log-probability of acceptance of input sequences x by a probabilistic DFA
    defined by the given initial state, transition, and accepting state probabilities.

    :param logit_initial_state_probs: (S,) tensor of logits for the initial state probabilities.
    :param logit_transition_probs: (S, C, S) tensor of logits for the transition probabilities.
    :param logit_accepting_state_probs: (S,) tensor of logits for the accepting state probabilities.
    :param log_input_probs: (N, L, C) tensor of log-probabilities of each input transition.
        Must logsumexp to < 0 along the C dimension.
    :return: (N,) tensor of log-probabilities of acceptance for each input sequence.
    """
    N, L, C = log_input_probs.shape
    [S] = logit_initial_state_probs.shape
    assert logit_transition_probs.shape == (S, C, S)
    assert logit_accepting_state_probs.shape == (S,)

    initial_state_probs = nn.functional.softmax(logit_initial_state_probs, dim=0)
    transition_probs = nn.functional.softmax(logit_transition_probs, dim=2)
    accepting_state_probs = nn.LogSigmoid()(logit_accepting_state_probs)
    input_probs = torch.exp(log_input_probs)

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
        log_p_state + accepting_state_probs.unsqueeze(0), dim=1
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

    def entropy(self, weight_initial=1.0, weight_transition=5.0, weight_accepting=1.0):
        """
        Computes the entropy of the PDFA's parameters.

        :param weight_initial: float, weight for the initial state probabilities entropy.
        :param weight_transition: float, weight for the transition probabilities entropy.
        :param weight_accepting: float, weight for the accepting state probabilities entropy.
        :return: float, weighted sum of entropies of the PDFA's parameters.
        """
        log_initial_probs = entropy_of_logits(self.logit_initial_state_probs)
        log_transition_probs = entropy_of_logits(
            self.logit_transition_probs, dim=2
        ).mean()
        log_accepting_probs = entropy_of_logits(self.logit_accepting_state_probs)
        return (
            weight_initial * log_initial_probs
            + weight_transition * log_transition_probs
            + weight_accepting * log_accepting_probs
        ) / (weight_initial + weight_transition + weight_accepting)

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

    def forward(self, log_input_probs):
        """
        Outputs (1, N) tensor of log-probabilities of acceptance for each input sequence.

        The 1 dimension is for forward compatibility if we decide to batch multiple PDFAs.
        """
        return pdfa(
            self.logit_initial_state_probs,
            self.logit_transition_probs,
            self.logit_accepting_state_probs,
            log_input_probs,
        )[None]


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
