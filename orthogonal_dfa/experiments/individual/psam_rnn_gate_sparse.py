import argparse

from orthogonal_dfa.experiments.gate_experiments import (
    get_starting_gates,
    train_rnn_psams_sparse,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for training.",
    )
    parser.add_argument(
        "--initial-threshold",
        type=float,
        help="Initial threshold.",
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
    parser.add_argument(
        "--build-on",
        choices=["nothing", "psam-linear-alt"],
        default="nothing",
        help="What to build the RNNPSAMProcessorNoise on top of.",
    )
    args = parser.parse_args()
    train_rnn_psams_sparse(
        args.seed,
        hidden_size=args.hidden_size,
        layers=args.layers,
        initial_threshold=args.initial_threshold,
        starting_gates=get_starting_gates(args.build_on),
    )


if __name__ == "__main__":
    main()
