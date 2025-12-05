from argparse import ArgumentParser

from orthogonal_dfa.experiments.gate_experiments import train_psam_linear, train_psamdfa


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for training.",
    )
    parser.add_argument(
        "--build-on",
        choices=["nothing", "psam_linear"],
        help="Whether to build on PSAM linear model.",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of PDFAs to train.",
    )

    parser.add_argument(
        "--num-states",
        type=int,
        default=4,
        help="Number of states for the PDFA.",
    )

    args = parser.parse_args()
    seed = args.seed

    if args.build_on == "psam_linear":
        gates, _, _ = train_psam_linear()
        baselines = list(gates)
    elif args.build_on == "nothing":
        baselines = []
    else:
        raise ValueError("Must specify --build-on")

    train_psamdfa(baselines, seed=seed, count=args.count, num_states=args.num_states)


if __name__ == "__main__":
    main()
