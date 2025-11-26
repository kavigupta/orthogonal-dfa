import copy
import io
from datetime import datetime

import numpy as np
import torch
from permacache import drop_if_equal, permacache, stable_hash

from orthogonal_dfa.oracle.evaluate import (
    batch_run,
    conditional_mutual_information_from_log_confusion,
    evaluate_pdfas,
    multidimensional_confusion_from_proabilistic_results,
)
from orthogonal_dfa.oracle.run_model import create_dataset
from orthogonal_dfa.psams.psams import TorchPSAMs, UnionedPSAMs, flip_log_probs


def cross_entropy_loss(psams_output, target):
    """
    Computes the xentropy loss between the PSAMs output and the target.

    :param psams_output: torch.Tensor of shape (batch_size, num_psams) representing log probabilities
    :param target: binary torch.Tensor of shape (batch_size,)
    :return: torch.Tensor, the computed cross-entropy loss.
    """
    loss = -(
        target * psams_output + (1 - target) * flip_log_probs(psams_output)
    )  # (batch_size, num_psams)
    return loss.mean()


@permacache(
    "orthogonal_dfa/psams/train/train_psams_3",
    key_function=dict(lr=drop_if_equal(3e-3)),
)
def train_psams(*, two_r, num_psams, seed, data_loader, num_batches, lr=3e-3):
    print(f"Train {num_psams=} {two_r=} {seed=}")
    torch.manual_seed(seed)
    psams = UnionedPSAMs(TorchPSAMs.create(two_r, 4, num_psams)).cuda()

    optim = torch.optim.Adam(psams.parameters(), lr=lr)
    losses = []
    for it in range(num_batches):
        x, y = data_loader.next(it)
        result = psams(x).cpu().squeeze(1)
        loss = cross_entropy_loss(result, y.float())
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if (it + 1) % 1000 == 0:
            print(f"Epoch {it+1}/{num_batches}, Loss: {loss.item():.4f}")
    return psams, losses


def identify_first_best_by_validation(rewards, tolerance):
    """
    Identifies the first epoch where the reward is within a certain tolerance of the best reward.

    :param rewards: List of float, the training rewards over epochs.
    :param tolerance: float, the tolerance level to consider a reward as "best".
    :return: int, the index of the first epoch with reward within tolerance of the best reward.
    """
    best_reward = max(rewards)
    for i, reward in enumerate(rewards):
        if reward >= best_reward - tolerance:
            return i
    return len(rewards) - 1  # Fallback to last epoch if none found


def create_training_dataset(
    exon, oracle, baseline_psam_pdfas, train_dataset_size, seed
):
    random, hard_target = create_dataset(
        exon,
        oracle,
        count=train_dataset_size,
        seed=int(stable_hash(("train-psam-pdfas", seed)), 16),
    )
    x = torch.eye(4)[random].cuda()
    with torch.no_grad():
        baseline_psam_pdfas = [batch_run(bp.cuda().eval(), x) for bp in baseline_psam_pdfas]
        assert all(bp.shape[0] == 1 for bp in baseline_psam_pdfas)
        baseline_psam_pdfas = [bp.squeeze(0) for bp in baseline_psam_pdfas]
    y = hard_target.float().cuda()
    y = torch.clamp(y, min=1e-7).log()
    return x, y, baseline_psam_pdfas


@permacache(
    "orthogonal_dfa/psams/train/train_psam_pdfa_full_learning_curve_6",
    key_function=dict(
        exon=stable_hash,
        oracle=stable_hash,
        starting_psam_pdfa=stable_hash,
        baseline_psam_pdfas=stable_hash,
        val_every=drop_if_equal(100),
        train_dataset_size=drop_if_equal(10_000),
    ),
)
def train_psam_pdfa_full_learning_curve(
    exon,
    oracle,
    starting_psam_pdfa,
    baseline_psam_pdfas,
    *,
    seed,
    epochs=1000,
    val_every=100,
    train_dataset_size=10_000,
    lr=1e-2,
    batch_size,
):
    m = copy.deepcopy(starting_psam_pdfa).cuda().train()
    opt = torch.optim.Adam(m.parameters(), lr)
    loss = []
    val_loss_epochs = []
    val_loss = []
    models = []
    for epoch in range(0, epochs):
        x, y, bpp = create_training_dataset(
            exon, oracle, baseline_psam_pdfas, train_dataset_size, (seed, epoch)
        )
        epoch_loss, m, opt = train_for_epoch(
            x, y, bpp, opt, m, batch_size=batch_size
        )
        m, opt = serialize_and_deserialize(m, opt)
        if (epoch + 1) % val_every == 0 or (epoch + 1 == epochs):
            validation_loss = conditional_mutual_information_from_log_confusion(
                evaluate_pdfas(
                    exon,
                    m,
                    baseline_psam_pdfas,
                    oracle,
                    seed=int(stable_hash(("val-psam-pdfa", seed)), 16),
                )
            )[0]
            val = f"; Val CMI: {validation_loss:.4f}"
            val_loss_epochs.append((epoch + 1) * train_dataset_size // batch_size)
            val_loss.append(validation_loss)
        else:
            val = ""
        print(
            f"{datetime.now()} Epoch {epoch + 1},"
            f" CMI: {np.mean(epoch_loss):.4f}" + val
        )

        loss.extend(epoch_loss)
        models.append(copy.deepcopy(m).eval().cpu())

    return models, dict(
        train_loss=loss, val_loss_epochs=val_loss_epochs, val_loss=val_loss
    )


def serialize_and_deserialize(model, opt):
    buffer = io.BytesIO()
    torch.save((model.state_dict(), opt.state_dict()), buffer)
    buffer.seek(0)
    model_sd, opt_sd = torch.load(buffer)
    model, opt = copy.deepcopy((model, opt))
    model.load_state_dict(model_sd)
    opt.load_state_dict(opt_sd)
    return model, opt


@permacache(
    "orthogonal_dfa/psams/train/train_for_epoch",
    key_function=dict(
        x=stable_hash,
        y=stable_hash,
        baseline_psam_pdfa_val=stable_hash,
        opt=lambda opt: stable_hash(opt.state_dict()),
        m=stable_hash,
    ),
)
def train_for_epoch(
    x,
    y,
    baseline_psam_pdfa_val,
    opt,
    m,
    *,
    batch_size,
):
    loss = []
    for start in range(0, len(x), batch_size):
        opt.zero_grad()
        batch_x = x[start : start + batch_size]
        batch_y = y[start : start + batch_size]
        batch_baseline_psam_pdfa_val = [
            bp[start : start + batch_size] for bp in baseline_psam_pdfa_val
        ]
        batch_yp = m(batch_x)
        mi = conditional_mutual_information_from_log_confusion(
            multidimensional_confusion_from_proabilistic_results(
                batch_y, batch_yp, batch_baseline_psam_pdfa_val
            )
        )
        (-mi).backward()
        opt.step()
        loss.append(mi.item())
    return loss, m, opt
