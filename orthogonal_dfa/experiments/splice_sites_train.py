import frame_alignment_checks as fac
from matplotlib import pyplot as plt

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.manual_dfa.splice_site_dfa import splice_site_psam_pdfa
from orthogonal_dfa.psams.train import (
    identify_first_best_by_validation,
    train_psam_pdfa_full_learning_curve,
)
from orthogonal_dfa.spliceai.load_model import load_spliceai

tolerance = 0.5e-3


def trained_splice_site(which, num_epochs):
    oracle = load_spliceai(400, 0).cuda()
    ms, meta = train_psam_pdfa_full_learning_curve(
        default_exon,
        oracle,
        splice_site_psam_pdfa(which, 0.5),
        [],
        seed=1,
        epochs=num_epochs,
        lr=1e-3,
    )
    m = ms[identify_first_best_by_validation(meta["val_loss"], tolerance=tolerance)]
    return m, meta


def plot_learning_curve(meta, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(meta["train_loss"], color=fac.plotting.line_color(0), label="Train")
    ax.plot(
        meta["val_loss_epochs"],
        meta["val_loss"],
        color=fac.plotting.line_color(1),
        label="Validation",
    )
    best_val = identify_first_best_by_validation(meta["val_loss"], tolerance)
    ax.scatter(
        meta["val_loss_epochs"][best_val],
        meta["val_loss"][best_val],
        color=fac.plotting.line_color(1),
        label="Selected Model",
    )
    ax.set_ylabel("Mutual information [b]")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(axis="y")
