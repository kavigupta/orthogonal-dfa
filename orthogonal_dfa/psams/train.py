import torch
from permacache import drop_if_equal, permacache

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
