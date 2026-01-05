import unittest

from dconstruct import construct
import numpy as np
import torch
from parameterized import parameterized

from orthogonal_dfa.module.sparsity.sparsity_layer import sparsity_types

eps = 1e-3

sparse_spec = dict(
    type="SparseLayerWithBatchNorm",
    underlying_sparsity_spec=dict(
        type="EnforceSparsityPerChannel1D",
        enforce_sparsity_per_channel_spec=dict(
            type="EnforceSparsityPerChannelAccumulated",
            accumulation_stop_strategy=dict(
                type="StopAtFixedNumberMotifs", num_motifs=10
            ),
        ),
    ),
    affine=False,
    input_dimensions=1,
)


class TestSparsityTargetsAutomatically(unittest.TestCase):
    @parameterized.expand([(seed,) for seed in range(5)])
    def test_basic(self, seed):
        rng = np.random.default_rng(seed)
        num_psams = rng.choice([4, 8, 16])
        sparsity = rng.uniform(0.05, 0.95)
        offset = torch.tensor(rng.normal(0.0, 5.0, size=(num_psams,))).float()
        sparse_layer = construct(
            sparsity_types(),
            sparse_spec,
            starting_sparsity=sparsity,
            channels=num_psams,
        )
        # charge the sparsity layer
        for _ in range(500):
            data = torch.randn(10, 100, num_psams) + offset
            sparse_layer(data)
        sparse_layer.eval()
        # check that the sparsity is close to target
        data = torch.randn(1000, 100, num_psams) + offset
        output = sparse_layer(data)
        actual_sparsity = (output == 0).float().mean().item()
        self.assertAlmostEqual(
            actual_sparsity,
            sparsity,
            delta=0.05,
            msg=f"Sparsity {actual_sparsity} not close to target {sparsity}",
        )
        # doesn't change in eval mode
        for _ in range(500):
            data = torch.randn(10, 100, num_psams) + offset + 100
            sparse_layer(data)
        data = torch.randn(1000, 100, num_psams) + offset
        output = sparse_layer(data)
        actual_sparsity = (output == 0).float().mean().item()
        self.assertAlmostEqual(
            actual_sparsity,
            sparsity,
            delta=0.05,
            msg=f"Sparsity {actual_sparsity} not close to target {sparsity} after more data",
        )

        # does chagne in train mode
        sparse_layer.train()
        for _ in range(500):
            data = torch.randn(10, 100, num_psams) + offset + 100
            sparse_layer(data)
        data = torch.randn(1000, 100, num_psams) + offset + 100
        sparse_layer.eval()
        output = sparse_layer(data)
        actual_sparsity = (output == 0).float().mean().item()
        self.assertAlmostEqual(
            actual_sparsity,
            sparsity,
            delta=0.05,
            msg=f"Sparsity {actual_sparsity} not close to target {sparsity} after training more",
        )
