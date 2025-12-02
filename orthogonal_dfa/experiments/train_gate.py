import copy
from datetime import datetime
from typing import List

import numpy as np
from render_psam import render_psam
import torch
from permacache import permacache, stable_hash
from torch import nn

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.module.residual_gate import ResidualGate
from orthogonal_dfa.oracle.run_model import create_dataset


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
    ),
)
def train(
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
            seed=int(stable_hash(("train", seed, epoch)), 16) % (2**32 - 1),
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
            y_pred = gate.compute_input(x, y_targ)
            eval_loss_control = loss(y, y_targ).item()
            eval_loss = loss(y, y_pred).item()
    finally:
        if is_training:
            gate.train()
    return eval_loss, eval_loss_control


def train_multiple(
    gates: List[ResidualGate],
    lr: float,
    exon: RawExon,
    oracle,
    *,
    seed,
    **kwargs,
):
    trained_gates = []
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
    ax_monotonic.plot(*gate.monotonic.plot_function(extra_range=0.5))
    ax_monotonic.set_xlabel("Internal response")
    ax_monotonic.set_ylabel("log-Bayes Odds")
