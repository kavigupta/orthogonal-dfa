import gc

import torch

from orthogonal_dfa.baseline import MonolithicLinearLayer, PSAMsFollowedByLinear
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.experiments.train_from_scratch import oracle
from orthogonal_dfa.experiments.train_gate import evaluate_multiple, train_multiple
from orthogonal_dfa.module.residual_gate import InputMonotonicModelingGate
from orthogonal_dfa.psams.psam_pdfa import PSAMPDFA
from orthogonal_dfa.utils.pdfa import PDFA


def _clear_tensors():
    gc.collect()
    torch.cuda.empty_cache()


def train_many(constructor, count, *, seed=0, starting_gates=(), epochs=500):
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
    gates_trained, loss = train_multiple(
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
    )

    _clear_tensors()

    results = evaluate_multiple(
        default_exon, oracle(), gates_trained, count=100_000, seed=0
    )
    results = results[:, 1] - results[:, 0]
    return gates_trained, loss, results


def train_mll(count=5):
    return train_many(lambda length: MonolithicLinearLayer(4, length), count)


def train_psam_linear(count=11):
    return train_many(lambda length: PSAMsFollowedByLinear(4, 1, 9, length), count)


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


def main():
    train_mll()
    pl, _, _ = train_psam_linear()
    for seed in range(10):
        train_psamdfa([], seed=seed)
        train_psamdfa(pl, seed=seed)


if __name__ == "__main__":
    main()
