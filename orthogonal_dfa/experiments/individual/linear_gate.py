from argparse import ArgumentParser

from orthogonal_dfa.experiments.gate_experiments import train_psam_linear_on_others


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--prev-count",
        type=int,
        default=11,
        help="Number of previous PSAM linear models to build on.",
    )
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    train_psam_linear_on_others(args.prev_count, args.seed)


if __name__ == "__main__":
    main()
