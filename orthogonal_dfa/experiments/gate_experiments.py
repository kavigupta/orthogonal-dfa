import gc

import torch

from orthogonal_dfa.baseline import MonolithicLinearLayer, PSAMsFollowedByLinear
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.experiments.train_from_scratch import oracle
from orthogonal_dfa.experiments.train_gate import (
    evaluate_multiple,
    train_multiple,
    train_multiple_with_alternates,
)
from orthogonal_dfa.module.residual_gate import InputMonotonicModelingGate
from orthogonal_dfa.module.rnn import RNNProcessor, RNNPSAMProcessorNoise
from orthogonal_dfa.psams.psam_pdfa import PSAMPDFA
from orthogonal_dfa.psams.psams import TorchPSAMs
from orthogonal_dfa.utils.pdfa import PDFA


def _clear_tensors():
    gc.collect()
    torch.cuda.empty_cache()


def train_many(
    constructor, count, *, seed=0, starting_gates=(), epochs=500, num_alternates=None
):
    torch.manual_seed(seed)
    gates = [
        InputMonotonicModelingGate(
            # MonolithicLinearLayer(
            #     num_input_channels=4, input_length=default_exon.random_text_length
            # ),
            constructor(default_exon.random_text_length),
            5,
            100,
        ).cuda()
        for _ in range(count)
    ]
    kwargs = {"num_alternates": num_alternates} if num_alternates is not None else {}
    gates_trained, loss = (
        train_multiple_with_alternates if num_alternates is not None else train_multiple
    )(
        gates,
        1e-4,
        default_exon,
        oracle(),
        epochs=epochs,
        batch_size=1000,
        train_count=100_000,
        seed=seed,
        do_not_train_phi=False,
        starting_gates=starting_gates,
        **kwargs,
    )

    _clear_tensors()

    results = evaluate_multiple(
        default_exon, oracle(), gates_trained, count=100_000, seed=0
    )
    results = results[:, 1] - results[:, 0]
    return gates_trained, loss, results


def train_mll(count=5):
    return train_many(lambda length: MonolithicLinearLayer(4, length), count)


def train_psam_linear_with_alternates(length):
    return train_many(
        lambda l: PSAMsFollowedByLinear(4, 1, 9, l),
        length,
        num_alternates=3,
        epochs=2000,
    )


def train_psam_linear(count=11):
    return train_many(lambda length: PSAMsFollowedByLinear(4, 1, 9, length), count)


def train_psam_linear_on_others(prev_count, seed):
    return train_many(
        lambda length: PSAMsFollowedByLinear(4, 1, 9, length),
        1,
        seed=seed,
        starting_gates=train_psam_linear(prev_count)[0],
        epochs=2000,
    )


def train_psamdfa(
    starting_gates, *, seed=0, count=1, num_states=4, pdfa_typ=PDFA, epochs=None
):
    if epochs is None:
        epochs = 2000 if not starting_gates else 4000
    return train_many(
        lambda length: PSAMPDFA.create(
            num_input_channels=4,
            num_psams=4,
            two_r=8,
            num_states=num_states,
            pdfa_typ=pdfa_typ,
        ),
        count,
        seed=seed,
        starting_gates=starting_gates,
        epochs=epochs,
    )


def train_rnn_direct(seed):
    return train_many(
        lambda length: RNNProcessor(num_inputs=4, hidden_size=100, num_layers=2).cuda(),
        1,
        seed=seed,
        epochs=2000,
    )


def train_rnn_psams(seed, neg_log_noise_level):
    return train_many(
        lambda length: RNNPSAMProcessorNoise(
            TorchPSAMs.create(two_r=8, channels=4, num_psams=4),
            RNNProcessor(num_inputs=4, hidden_size=100, num_layers=2).cuda(),
            noise_level=10 ** (-neg_log_noise_level),
        ),
        1,
        seed=seed,
        epochs=2000,
    )


def main():
    train_mll()
    pl, _, _ = train_psam_linear()
    for seed in range(10):
        train_psamdfa([], seed=seed)
        train_psamdfa(pl, seed=seed)


if __name__ == "__main__":
    main()
