import copy
import io
import math
from datetime import datetime
from typing import List

import numpy as np
import torch
from matplotlib import gridspec
from matplotlib import pyplot as plt
from permacache import drop_if_equal, permacache, stable_hash
from render_psam import render_psam
from torch import nn

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.module.residual_gate import ResidualGate
from orthogonal_dfa.oracle.run_model import create_dataset
from orthogonal_dfa.utils.pdfa import to_dfa_for_viz


def training_data(
    exon: RawExon,
    oracle: nn.Module,
    prev_gates: List[ResidualGate],
    *,
    count: int,
    seed: int,
    device=torch.device("cpu"),
):
    x, y = create_dataset(exon, oracle, count=count, seed=seed)
    y = y.to(device)
    y_targ = torch.zeros(y.shape[0], device=device)
    x = torch.eye(4, device=device)[x]
    for gate in prev_gates:
        assert (
            not gate.training
        ), "Prior gates must be in eval mode during data generation"
        with torch.no_grad():
            y_targ = gate.compute_input(x, y_targ)
    return x, y, y_targ


def loss(y, y_targ):
    bce = nn.BCEWithLogitsLoss()
    # negate this because we are looking at this as the residual at the end of the model
    return bce(-y_targ, y.float()) / np.log(2)


@permacache(
    "orthogonal_dfa/expriments/train_gate/train_2",
    key_function=dict(
        gate=stable_hash,
        exon=stable_hash,
        oracle=stable_hash,
        prev_gates=lambda x: tuple(stable_hash(g) for g in x),
        start_epoch=drop_if_equal(0),
    ),
    multiprocess_safe=True,
)
def train_direct(
    gate: ResidualGate,
    lr: float,
    exon: RawExon,
    oracle,
    prev_gates: List[ResidualGate],
    *,
    epochs: int,
    batch_size: int,
    train_count: int,
    new_data_every_epoch: int = 5,
    seed: int,
    do_not_train_phi: bool,
    start_epoch: int = 0,
):
    gate = copy.deepcopy(gate)
    gate.train()
    assert gate.training, "Gate must be in training mode"
    optimizer = torch.optim.Adam(gate.parameters(), lr=lr)

    def td(epoch):
        return training_data(
            exon,
            oracle,
            prev_gates,
            count=train_count,
            seed=int(stable_hash(("train", seed, epoch + start_epoch)), 16)
            % (2**32 - 1),
            device=next(gate.parameters()).device,
        )

    results = []

    x, y, y_targ = td(0)
    for epoch in range(epochs):
        if (
            epoch != 0
            and new_data_every_epoch is not None
            and epoch % new_data_every_epoch == 0
        ):
            x, y, y_targ = td(epoch)
        epoch_loss = train_for_an_epoch(
            gate,
            optimizer,
            data=(x, y, y_targ),
            batch_size=batch_size,
            do_not_train_phi=do_not_train_phi,
        )
        print(
            f"{datetime.now().isoformat()} Epoch {epoch+1}/{epochs}, Loss: {np.mean(epoch_loss):.4f}"
        )
        results.append(epoch_loss)
    return gate, results


def train(
    gate: ResidualGate,
    *args,
    epochs,
    **kwargs,
):
    all_losses = []
    for epoch_chunk in range(0, epochs, 500):
        gate, losses = train_direct(
            gate,
            *args,
            epochs=min(500, epochs - epoch_chunk),
            start_epoch=epoch_chunk,
            **kwargs,
        )
        all_losses.extend(losses)
    return gate, all_losses


def train_for_an_epoch(
    gate: ResidualGate,
    optimizer: torch.optim.Optimizer,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
    *,
    do_not_train_phi: bool,
):
    gate.train()
    losses = []
    x, y, y_targ = data
    for start in range(0, x.shape[0], batch_size):
        optimizer.zero_grad()
        end = min(start + batch_size, x.shape[0])
        x_batch = x[start:end]
        y_batch = y[start:end]
        y_targ_batch = y_targ[start:end]
        y_pred_batch = gate.compute_input(
            x_batch, y_targ_batch, do_not_train_phi=do_not_train_phi
        )
        batch_loss = loss(y_batch, y_pred_batch) - loss(y_batch, y_targ_batch)
        losses.append(batch_loss.item())
        batch_loss.backward()
        optimizer.step()

    return losses


@permacache(
    "orthogonal_dfa/experiments/train_gate/evaluate_2",
    key_function=dict(
        exon=stable_hash,
        oracle=stable_hash,
        prev_gates=lambda x: tuple(stable_hash(g) for g in x),
        gate=stable_hash,
    ),
)
def evaluate(
    exon: RawExon,
    oracle,
    prev_gates,
    gate,
    *,
    count: int,
    seed: int,
):
    is_training = gate.training
    gate.eval()
    try:
        x, y, y_targ = training_data(
            exon,
            oracle,
            prev_gates,
            count=count,
            seed=int(stable_hash(("validation", seed)), 16) % (2**32 - 1),
            device=next(gate.parameters()).device,
        )
        with torch.no_grad():
            y_pred = compute_input_batched(gate, x, y_targ, batch_size=1000)
            eval_loss_control = loss(y, y_targ).item()
            eval_loss = loss(y, y_pred).item()
    finally:
        if is_training:
            gate.train()
    return eval_loss, eval_loss_control


def compute_input_batched(
    gate: ResidualGate,
    x: torch.Tensor,
    y_targ: torch.Tensor,
    batch_size: int,
):
    assert not gate.training, "Gate must be in eval mode"
    with torch.no_grad():
        assert x.shape[0] == y_targ.shape[0]
        results = [
            gate.compute_input(x[i : i + batch_size], y_targ[i : i + batch_size])
            for i in range(0, x.shape[0], batch_size)
        ]
    return torch.cat(results, dim=0)


def train_multiple(
    gates: List[ResidualGate],
    lr: float,
    exon: RawExon,
    oracle,
    *,
    starting_gates=(),
    seed,
    **kwargs,
):
    trained_gates = [*starting_gates]
    all_losses = []
    for i, gate in enumerate(gates):
        print(f"Training gate {i+1}/{len(gates)}")
        gate, losses = train(
            gate,
            lr,
            exon,
            oracle,
            trained_gates,
            **kwargs,
            seed=int(stable_hash((seed, i)), 16) % (2**32 - 1),
        )
        trained_gates.append(gate.eval())
        all_losses.append(losses)
    return trained_gates, all_losses


def evaluate_multiple(
    exon: RawExon,
    oracle,
    gates: List[ResidualGate],
    *,
    seed,
    **kwargs,
):
    eval_losses = []
    for i, gate in enumerate(gates):
        eval_loss, eval_loss_control = evaluate(
            exon,
            oracle,
            gates[:i],
            gate,
            **kwargs,
            seed=int(stable_hash((seed, i)), 16) % (2**32 - 1),
        )
        eval_losses.append((eval_loss, eval_loss_control))
    return np.array(eval_losses)


def plot_linear_psam_gate(gate, *axs):
    ax_psam, ax_response, ax_monotonic = axs

    render_psam(gate.phi.psams.sequence_logos[0], psam_mode="raw", ax=ax_psam)
    ax_psam.set_xticks([])
    ax_response.plot(gate.phi.linear.linear.weight[0].detach().cpu().numpy())
    ax_response.set_xlabel("Position in exon")
    ax_response.set_ylabel("Internal response level")
    plot_monotonicity(gate.monotonic, ax_monotonic)


def plot_monotonicity(monotonic, ax_monotonic):
    xmon, ymon = monotonic.plot_function(extra_range=0.5)
    ax_monotonic.plot(xmon, ymon)
    ax_monotonic.set_xlabel("Model log-probability")
    ax_monotonic.set_ylabel("log-Bayes Odds")


def plot_pdfa(gate):
    p = gate.phi.psam
    size = 3
    num_psams = len(p.sequence_logos)

    # Calculate grid dimensions for PSAMs (arrange side by side)
    psam_cols = min(num_psams, 4)  # Max 4 columns, adjust as needed
    psam_rows = math.ceil(num_psams / psam_cols)

    height_ratios = [1] * psam_rows + [5, 1]
    fig = plt.figure(figsize=(size * 2 * psam_cols, (psam_rows + 3) * size))
    gs = gridspec.GridSpec(
        psam_rows + 2,
        psam_cols,
        figure=fig,
        hspace=0.3,
        wspace=0.3,
        height_ratios=height_ratios,
    )

    # Plot PSAMs in grid at the top
    for i in range(num_psams):
        row = i // psam_cols
        col = i % psam_cols
        ax_psam = fig.add_subplot(gs[row, col])
        render_psam(p.sequence_logos[i], psam_mode="raw", ax=ax_psam)
        ax_psam.set_xticks([])

    # PDFA diagram - full width below PSAMs
    pdfa = to_dfa_for_viz(gate.phi.pdfa.initialized, 0.1)
    graph = pdfa.show_diagram()
    ax_pdfa = fig.add_subplot(gs[psam_rows, :])
    ax_pdfa.imshow(plt.imread(io.BytesIO(graph.draw(format="png"))))
    ax_pdfa.axis("off")
    ax_pdfa.set_title("PDFA Diagram")

    # Monotonicity plot - full width at the bottom
    ax_monotonic = fig.add_subplot(gs[psam_rows + 1, :])
    plot_monotonicity(gate.monotonic, ax_monotonic)
