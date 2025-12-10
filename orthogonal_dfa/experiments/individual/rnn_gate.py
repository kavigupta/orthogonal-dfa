# pylint: disable=duplicated-code
import argparse

from orthogonal_dfa.experiments.gate_experiments import train_rnn_direct


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for training.",
    )
    args = parser.parse_args()
    train_rnn_direct(args.seed)


if __name__ == "__main__":
    main()
