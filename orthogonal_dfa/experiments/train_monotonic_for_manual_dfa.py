import copy

import torch

from orthogonal_dfa.experiments.train_gate import evaluate, train
from orthogonal_dfa.module.residual_gate import InputMonotonicModelingGate
from orthogonal_dfa.utils.pdfa import PDFAOutputtingProbabilities


def train_monotonic_for_manual_dfa(
    exon,
    test_dfa,
    control_dfas,
    oracle,
    *,
    seed=0,
    max_z_score=4.0,
    num_input_breaks=1000,
    exp_probs=True,
):
    control_gates = []
    if control_dfas:
        *earlier, last = control_dfas
        earlier, last = train_monotonic_for_manual_dfa(
            exon,
            last,
            earlier,
            oracle,
            seed=seed,
            max_z_score=max_z_score,
            num_input_breaks=num_input_breaks,
            exp_probs=exp_probs,
        )
        control_gates = [*earlier, last.eval()]
    torch.manual_seed(seed)

    if exp_probs:
        test_dfa = PDFAOutputtingProbabilities(test_dfa)

    gate = InputMonotonicModelingGate(test_dfa, max_z_score, num_input_breaks).cuda()
    gate, _ = train(
        gate,
        1e-2,
        exon,
        oracle,
        control_gates,
        epochs=10,
        batch_size=1000,
        train_count=100_000,
        seed=seed,
        do_not_train_phi=True,
    )
    return control_gates, copy.deepcopy(gate)


def evaluate_monotonic_for_manual_dfa(exon, test_dfa, control_dfas, oracle, **kwargs):
    control_gates, gate = train_monotonic_for_manual_dfa(
        exon, test_dfa, control_dfas, oracle, **kwargs
    )
    curr, prev = evaluate(
        exon, oracle, control_gates, gate.eval(), count=100_000, seed=0
    )
    return prev - curr
