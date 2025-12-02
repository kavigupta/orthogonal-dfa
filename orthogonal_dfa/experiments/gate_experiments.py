import torch

from orthogonal_dfa.baseline import MonolithicLinearLayer, PSAMsFollowedByLinear
from orthogonal_dfa.experiments.train_from_scratch import oracle
from orthogonal_dfa.experiments.train_gate import evaluate_multiple, train_multiple
from orthogonal_dfa.module.residual_gate import InputMonotonicModelingGate
from orthogonal_dfa.data.exon import default_exon


def train_many(constructor, count):
    torch.manual_seed(0)
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
        epochs=500,
        batch_size=1000,
        train_count=100_000,
        seed=0,
        do_not_train_phi=False,
    )
    results = evaluate_multiple(
        default_exon, oracle(), gates_trained, count=100_000, seed=0
    )
    results = results[:, 1] - results[:, 0]
    return gates_trained, loss, results


def train_mll(count=5):
    return train_many(lambda length: MonolithicLinearLayer(4, length), count)


def train_psam_linear(count=100):
    return train_many(lambda length: PSAMsFollowedByLinear(4, 1, 9, length), count)


def main():
    train_mll()
    train_psam_linear()


if __name__ == "__main__":
    main()
