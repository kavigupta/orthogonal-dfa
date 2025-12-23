import argparse

from orthogonal_dfa.experiments.gate_experiments import (
    get_starting_gates,
    train_rnn_psams,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for training.",
    )
    parser.add_argument(
        "--neg-log-noise-level",
        type=float,
        help="Negative log noise level for RNNPSAMProcessorNoise.",
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
    train_rnn_psams(
        args.seed,
        args.neg_log_noise_level,
        hidden_size=args.hidden_size,
        layers=args.layers,
        starting_gates=get_starting_gates(args.build_on),
    )


if __name__ == "__main__":
    main()
