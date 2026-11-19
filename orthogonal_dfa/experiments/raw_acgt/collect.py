"""Every seed's residual stages as a table; writes data/neural/raw_acgt/residual_products.json.

python -m orthogonal_dfa.experiments.raw_acgt.collect"""

import json

from .common import DATA, RESIDUAL, SEEDS


def show(r):
    if r["stage"] == 1:
        print(
            f"s{r['seed']} stage 1: {r['stage_states']:3d} st | product "
            f"{r['product_states']:3d} min phi {r['product_phi']:.3f}"
        )
        return
    frames = " ".join(f"{k} {v:+.2f}" for k, v in r["stage_frames"].items())
    print(
        f"s{r['seed']} stage {r['stage']}: {r['stage_states']:3d} st residual phi "
        f"{r['residual_phi']:.3f} (flat {r['stage_flat_phi']:+.3f}) | product "
        f"{r['reachable']:3d} reach {r['product_states']:3d} min phi "
        f"{r['product_phi']:.3f} | {r['hours']:.2f} h | frames {frames}"
    )
    restarts = " ".join(
        f"r{x['restart']}:{x['states']}st/{x['residual_phi']:.3f}/{x['ending']}"
        for x in r["restarts"]
    )
    print(
        "      stage tracks:",
        ", ".join(f"{n} {v:+.2f}" for n, v in r["stage_features"]),
        "| product:",
        ", ".join(f"{n} {v:+.2f}" for n, v in r["product_features"][:3]),
        "| restarts:",
        restarts,
    )


def main():
    out = []
    for s in SEEDS:
        with open(RESIDUAL / f"s{s}" / "stages.jsonl") as f:
            for line in f:
                r = json.loads(line)
                show(r)
                out.append(
                    dict(
                        seed=s,
                        stage=r["stage"],
                        restart=r["restart"],
                        stage_dfa=r["dfa"],
                        product=r["product"],
                        product_phi=r["product_phi"],
                        residual_phi=r.get("residual_phi"),
                    )
                )
    with open(DATA / "residual_products.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
