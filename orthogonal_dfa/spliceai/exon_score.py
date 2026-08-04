"""The SpliceAI exon score, shared by the dataset pipeline and the E-L* oracle.

`oracle.run_model` scores full-length windows straight out of `data.sample_text`;
`l_star.examples.spliceai_oracle` scores ragged middles wrapped in the same
flanks.  Both read the same two positions out of the same model and feed it the
same one-hot encoding, so the score and the input encoding live here rather than
in either caller.
"""

import torch
import torch.nn.functional as F
from torch import nn

# Each flank keeps this many bases beyond the model's cl/2 half-context (matching
# data.sample_text's trim_zone = cl//2 + 2), so the output spans len(middle) +
# 2*FLANK_MARGIN positions: the acceptor is the first, the donor the last.
FLANK_MARGIN = 2


def device_of(model, device=None):
    return (
        torch.device(device) if device is not None else next(model.parameters()).device
    )


def one_hot(wrapped, *, device):
    """The float one-hot encoding of the base-index array ``wrapped``."""
    # pylint: disable=not-callable
    return F.one_hot(torch.as_tensor(wrapped, device=device), 4).float()


def forward_batch(model, wrapped, *, device):
    """One-hot ``wrapped`` and run ``model`` over it under no_grad.

    Checked on every forward rather than once at setup: the padding argument in
    wrap_with_flanks holds only in eval mode, and a caller sharing the model with
    a training loop can put it back into train mode at any point."""
    assert not model.training, (
        "model must be in eval mode: in train mode BatchNorm normalizes over the "
        "batch, so padded rows leak into every other row's score"
    )
    with torch.no_grad():
        return model(one_hot(wrapped, device=device))


def full_lengths(logits):
    """The middle lengths of a batch whose rows are all full length."""
    return torch.full(
        (len(logits),), logits.shape[1] - 2 * FLANK_MARGIN, device=logits.device
    )


def assert_output_width(output_width: int, expected: int):
    """Guard that the model's cl matches the exon the input was cut for.

    A model that trims a different cl shifts every output position, so the donor
    gets read off the wrong one and the score comes back as plausible noise."""
    assert output_width == expected, (
        f"model output width {output_width} does not match the expected "
        f"{expected}; the exon's cl and the model's cl probably disagree"
    )


def spliceai_exon_scores(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Exon score per sequence: mean of the acceptor log probability at the first
    output position and the donor log probability at the middle's last."""
    assert_output_width(logits.shape[1], int(lengths.max()) + 2 * FLANK_MARGIN)
    lyp = logits.log_softmax(-1)
    rows = torch.arange(len(lyp), device=lyp.device)
    acc = lyp[rows, 0, 1]
    don = lyp[rows, lengths + 2 * FLANK_MARGIN - 1, 2]
    return torch.stack([acc, don], -1).mean(-1)


class SpliceAIExonScore(nn.Module):
    """Wraps a splice model so its forward maps (one-hot ``x``, middle ``lengths``)
    to the per-sequence exon score; a composition-residual model can wrap this in
    turn and adjust the scalar it returns."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, lengths):
        return spliceai_exon_scores(self.model(x), lengths)
