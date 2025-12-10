import argparse

from orthogonal_dfa.experiments.gate_experiments import train_rnn_psams


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
    args = parser.parse_args()
    train_rnn_psams(args.seed, args.neg_log_noise_level)


if __name__ == "__main__":
    main()
