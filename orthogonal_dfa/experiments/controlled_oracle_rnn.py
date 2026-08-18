"""Train the sparse-PSAM RNN on the composition-deconfounded ("controlled") oracle.

The registered model ``r_rnn_500_1l_sparse_10_psams`` trains a sparse-PSAM RNN to
predict **raw SpliceAI**'s exon call.  This module trains the *same* architecture,
with the *same* sparse-gate schedule and training loop, against a different label:
the **controlled oracle** -- SpliceAI's exon score with a monotonic bag-of-k-mers
composition gate subtracted and median-thresholded
(:func:`orthogonal_dfa.l_star.examples.gate_composition_residual.gate_residual_oracle`).

The question the experiment asks: with composition regressed out, what motifs does a
sparse-PSAM model put its 10 kernels on -- splice-site / positional structure, frame /
stop-codon structure, or nothing coherent?

Only the *label source* changes.  Everything downstream of ``(x, y)`` -- the sparse
gate, the RNN, the BCE-in-bits loss, the epoch loop, the adaptive-sparsity schedule --
is reused unchanged from the SpliceAI pipeline (``train_gate.train_for_an_epoch`` etc.).
The loop is duplicated here (rather than reusing ``train_gate.train_direct``) only
because ``create_dataset`` is hard-wired to a SpliceAI-shaped model, so the controlled
oracle cannot be threaded through it; the core training code is left untouched so its
permacaches stay valid.
"""

import copy
from datetime import datetime
from functools import lru_cache

import numpy as np
import torch
from permacache import permacache, stable_hash

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.data.sample_text import sample_text
from orthogonal_dfa.experiments.gate_experiments import get_asl
from orthogonal_dfa.experiments.train_gate import train_for_an_epoch
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.module.residual_gate import InputMonotonicModelingGate
from orthogonal_dfa.module.rnn import RNNProcessor, RNNPSAMProcessorSparse
from orthogonal_dfa.psams.psams import TorchPSAMs
from orthogonal_dfa.spliceai.load_model import load_spliceai

#: Full random-middle length the oracle scores (== default_exon.random_text_length).
CONTROLLED_LENGTH = default_exon.random_text_length  # 189


@lru_cache(maxsize=1)
def controlled_oracle(length=CONTROLLED_LENGTH):
    """The composition-deconfounded SpliceAI oracle over full-length middles.

    Cached per process: fitting the composition gate is expensive and deterministic,
    so every labelling call reuses one fitted oracle.
    """
    return gate_residual_oracle(default_exon, load_spliceai(400, 0), length=length)


@permacache(
    "orthogonal_dfa/experiments/controlled_oracle_rnn/controlled_labels_v1",
    multiprocess_safe=True,
)
def _controlled_labels(count, seed):
    """Bit-packed boolean controlled-oracle labels for ``count`` random middles.

    The middles are exactly ``sample_text(default_exon, seed, count)`` -- identical to
    the SpliceAI pipeline's -- so only the *label* differs from ``create_dataset``.
    Packed to bits (like ``create_dataset_just_output``) to keep the cache small.
    """
    random, _ = sample_text(default_exon, seed, count)
    oracle = controlled_oracle()
    labels = oracle.membership_queries([row.tolist() for row in random])
    return np.packbits(np.asarray(labels, dtype=bool))


def controlled_training_data(count, seed, device):
    """``(x_onehot, y, y_targ)`` for the controlled oracle -- mirror of
    :func:`train_gate.training_data` with an empty ``prev_gates`` (so ``y_targ`` is 0)."""
    random, _ = sample_text(default_exon, seed, count)
    y_bits = _controlled_labels(count, seed)
    y = np.unpackbits(y_bits).astype(bool)[:count]
    x = torch.eye(4, device=device)[torch.tensor(random, device=device)]
    y = torch.tensor(y, device=device)
    y_targ = torch.zeros(y.shape[0], device=device)
    return x, y, y_targ


def build_sparse_psam_gate(*, hidden_size, layers, num_psams, initial_threshold, seed):
    """The exact gate ``train_rnn_psams_sparse`` builds: a sparse-PSAM RNN phi wrapped
    in an input-monotonic residual gate.  ``torch.manual_seed(seed)`` first, matching
    ``train_many``."""
    torch.manual_seed(seed)
    phi = RNNPSAMProcessorSparse(
        TorchPSAMs.create(two_r=8, channels=4, num_psams=num_psams),
        RNNProcessor(
            num_inputs=num_psams, hidden_size=hidden_size, num_layers=layers
        ).cuda(),
        asl=get_asl(num_psams, initial_threshold=initial_threshold),
    )
    return InputMonotonicModelingGate(phi, 5, 100, batch_norm=True).cuda()


@permacache(
    "orthogonal_dfa/experiments/controlled_oracle_rnn/train_chunk_v1",
    key_function=dict(gate=lambda g: stable_hash(g, version=2)),
    multiprocess_safe=True,
)
def _train_chunk(
    gate,
    *,
    epochs,
    start_epoch,
    seed,
    lr,
    batch_size,
    train_count,
    new_data_every_epoch,
    do_not_train_phi,
    notify_epoch_loss,
):
    """One ``<=500``-epoch training chunk -- a controlled-data copy of
    ``train_gate.train_direct``.  Chunked + permacached so a long run is resumable and
    each cache entry is bounded, exactly as ``train_gate.train`` does it."""
    gate = copy.deepcopy(gate)
    gate.train()
    optimizer = torch.optim.Adam(gate.parameters(), lr=lr)
    device = next(gate.parameters()).device

    def td(epoch):
        s = int(stable_hash(("train", seed, epoch + start_epoch)), 16) % (2**32 - 1)
        return controlled_training_data(train_count, s, device)

    results = []
    x, y, y_targ = td(0)
    for epoch in range(epochs):
        if epoch != 0 and new_data_every_epoch and epoch % new_data_every_epoch == 0:
            x, y, y_targ = td(epoch)
        epoch_loss = train_for_an_epoch(
            gate,
            optimizer,
            data=(x, y, y_targ),
            batch_size=batch_size,
            do_not_train_phi=do_not_train_phi,
        )
        epoch_full = epoch + start_epoch + 1
        print(
            f"{datetime.now().isoformat()} Epoch {epoch_full}, "
            f"Loss: {np.mean(epoch_loss):.4f}"
        )
        to_save = epoch_loss
        if epoch != epochs - 1 and notify_epoch_loss:
            gate_backup = copy.deepcopy(gate)
            sparse_info = gate.notify_epoch_loss(epoch_full, epoch_loss)
            if sparse_info is not None:
                to_save = dict(loss=epoch_loss, sparse_info=sparse_info)
                if sparse_info["keep_old_model"]:
                    to_save["gate_before_update"] = gate_backup
        results.append(to_save)
    return gate, results


def train_sparse_psams_controlled(
    seed=0,
    *,
    hidden_size=500,
    layers=1,
    num_psams=10,
    initial_threshold=0.60,
    epochs=20_000,
    finetune_epochs=50,
    lr=1e-5,
    batch_size=1000,
    train_count=100_000,
):
    """Train the sparse-PSAM RNN against the controlled oracle and return
    ``(gate, results)``.

    Hyperparameters default to ``r_rnn_500_1l_sparse_10_psams``'s (500 units, 1 layer,
    10 PSAMs, ``initial_threshold=0.60``, 20k epochs, lr 1e-5, 100k samples/epoch, fresh
    data every 5 epochs, 50 finetune epochs with the PSAMs frozen).  ``results`` is the
    per-epoch list ``train_direct`` produces; entries where the sparse schedule kept the
    pre-update model carry a ``"gate_before_update"`` snapshot -- the object the display
    notebook reads the final PSAMs from.
    """
    gate = build_sparse_psam_gate(
        hidden_size=hidden_size,
        layers=layers,
        num_psams=num_psams,
        initial_threshold=initial_threshold,
        seed=seed,
    )
    all_results = []
    for start in range(0, epochs, 500):
        gate, res = _train_chunk(
            gate,
            epochs=min(500, epochs - start),
            start_epoch=start,
            seed=seed,
            lr=lr,
            batch_size=batch_size,
            train_count=train_count,
            new_data_every_epoch=5,
            do_not_train_phi=False,
            notify_epoch_loss=True,
        )
        all_results.extend(res)
    if finetune_epochs:
        # Finetune with the PSAMs frozen (do_not_train_phi=True): finalises the
        # monotonic calibration without moving the learned kernels, mirroring
        # train_many(finetune_epochs=50).
        gate, res = _train_chunk(
            gate,
            epochs=finetune_epochs,
            start_epoch=epochs,
            seed=seed,
            lr=lr,
            batch_size=batch_size,
            train_count=train_count,
            new_data_every_epoch=5,
            do_not_train_phi=True,
            notify_epoch_loss=False,
        )
        all_results.extend(res)
    return gate, all_results


def final_sparse_gate(results, fallback):
    """The last ``gate_before_update`` snapshot in ``results`` (the final sparse gate),
    or ``fallback`` if the schedule never kept one."""
    snaps = [
        r["gate_before_update"]
        for r in results
        if isinstance(r, dict) and r.get("gate_before_update") is not None
    ]
    return snaps[-1] if snaps else fallback
