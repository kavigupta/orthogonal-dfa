from argparse import ArgumentParser

from orthogonal_dfa.experiments.gate_experiments import train_psam_linear, train_psamdfa
from orthogonal_dfa.utils.pdfa import PDFA, PDFAHyberbolicParameterization


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

    parser.add_argument(
        "--pdfa-typ",
        choices=["PDFA", "PDFAHyberbolicParameterization"],
        default="PDFA",
        help="Type of PDFA to use.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train each PDFA for.",
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

    pdfa_typ = {
        "PDFA": PDFA,
        "PDFAHyberbolicParameterization": PDFAHyberbolicParameterization,
    }[args.pdfa_typ]

    train_psamdfa(
        baselines,
        seed=seed,
        count=args.count,
        num_states=args.num_states,
        pdfa_typ=pdfa_typ,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
