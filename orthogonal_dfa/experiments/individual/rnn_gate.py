# pylint: disable=duplicate-code
import argparse

from orthogonal_dfa.experiments.gate_experiments import train_rnn_direct
from orthogonal_dfa.module.rnn import LSTMProcessor, RNNProcessor


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for training.",
    )
    parser.add_argument(
        "--constructor",
        type=str,
        choices=["RNNProcessor", "LSTMProcessor"],
        default="RNNProcessor",
        help="Type of RNN processor to use.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=100,
        help="Hidden size of the RNN.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of layers in the RNN.",
    )
    args = parser.parse_args()
    constructor = dict(RNNProcessor=RNNProcessor, LSTMProcessor=LSTMProcessor)[
        args.constructor
    ]
    train_rnn_direct(
        args.seed,
        constructor=constructor,
        hidden_size=args.hidden_size,
        layers=args.layers,
    )


if __name__ == "__main__":
    main()
