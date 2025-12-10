import copy

import torch

from orthogonal_dfa.psams.psam_pdfa import PSAMPDFA
from orthogonal_dfa.spliceai.best_psams_for_lssi import train_psams_for_splice_site
from orthogonal_dfa.utils.pdfa import PDFA
from orthogonal_dfa.utils.probability import ZeroProbability


def splice_site_pdfa(num_psams, p_t, zero_prob: ZeroProbability):
    """
    Creates a splice site PDFA given the number of PSAMs and the transition probability p_t.

    The PDFA has 2 states: an initial state and an accepting state. The transition from
    the initial state to the accepting state occurs with probability p_t upon reading
    a PSAM. If no PSAM is detected, the PDFA remains in the initial state.
    """
    initial_state_probs = torch.tensor(
        [1 - zero_prob.probability, zero_prob.probability]
    )
    transition_probs = torch.zeros((2, num_psams + 1, 2)) + zero_prob.probability
    # 0 transition maps every state to itself with probability 1
    transition_probs[torch.arange(2), 0, torch.arange(2)] = 1 - zero_prob.probability
    # 1-num_psams transitions map initial state to accepting state with probability p_t
    transition_probs[0, 1:, 1] = p_t
    transition_probs[0, 1:, 0] = 1 - p_t
    # Accepting state remains accepting
    transition_probs[1, :, 1] = 1 - zero_prob.probability
    accepting_state_probs = torch.tensor(
        [zero_prob.probability, 1 - zero_prob.probability]
    )
    return PDFA(
        torch.logit(initial_state_probs),
        torch.logit(transition_probs),
        torch.logit(accepting_state_probs),
    )


def splice_site_psam_pdfa(which, logit_p_t):
    logit_p_t = torch.clamp(torch.tensor(logit_p_t), -10, 10)
    p_t = torch.sigmoid(logit_p_t).item()
    psams = (
        train_psams_for_splice_site(which, 4, num_batches=100_000, seed=0, lr=3e-4)[0]
        .eval()
        .psams
    )
    psams = copy.deepcopy(psams).cuda()
    return PSAMPDFA(
        psams,
        splice_site_pdfa(4, p_t, ZeroProbability(1e-7)),
    ).cuda()

