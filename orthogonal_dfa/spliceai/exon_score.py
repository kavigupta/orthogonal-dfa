"""The SpliceAI exon score, shared by the dataset pipeline and the E-L* oracle.

`oracle.run_model` scores full-length windows straight out of `data.sample_text`;
`l_star.examples.spliceai_oracle` scores ragged middles wrapped in the same
flanks.  Both read the same two positions out of the same model, so the readout
and the batched forward live here rather than in either caller.
"""

import torch
import torch.nn.functional as F

# Each flank keeps this many bases beyond the model's cl/2 half-context (matching
# data.sample_text's trim_zone = cl//2 + 2), so the output spans len(middle) +
# 2*FLANK_MARGIN positions: the acceptor is the first, the donor the last.
FLANK_MARGIN = 2


def device_of(model, device=None):
    return (
        torch.device(device) if device is not None else next(model.parameters()).device
    )


def forward_batch(model, wrapped, *, device):
    """One-hot ``wrapped`` and run ``model`` over it under no_grad."""
    # pylint: disable=not-callable
    x = F.one_hot(torch.as_tensor(wrapped, device=device), 4).float()
    with torch.no_grad():
        return model(x)


def full_lengths(logits):
    """The middle lengths of a batch whose rows are all full length."""
    return torch.full(
        (len(logits),), logits.shape[1] - 2 * FLANK_MARGIN, device=logits.device
    )


def spliceai_exon_scores(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Exon score per sequence: mean of the acceptor log probability at the first
    output position and the donor log probability at the middle's last."""
    # The output width pins down the model's context length, so this catches a
    # model whose cl disagrees with the exon the flanks were cut from -- which
    # would otherwise just read the donor off the wrong position.
    assert logits.shape[1] == int(lengths.max()) + 2 * FLANK_MARGIN, (
        f"model output width {logits.shape[1]} does not match the widest middle "
        f"({int(lengths.max())}) plus 2*{FLANK_MARGIN}; the exon's cl and the "
        f"model's cl probably disagree"
    )
    lyp = logits.log_softmax(-1)
    rows = torch.arange(len(lyp), device=lyp.device)
    acc = lyp[rows, 0, 1]
    don = lyp[rows, lengths + 2 * FLANK_MARGIN - 1, 2]
    return torch.stack([acc, don], -1).mean(-1)
