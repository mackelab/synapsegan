import unittest
import pytest

import torch

from syngan.models.ojas_net import OjaNet
from syngan.models.update_rules import PolynomialRule


class TestNeuralNets(unittest.TestCase):
    """Test setup for neural nets."""

    n_presyn_neur = 3
    n_postsyn_neur = 1
    poly_rule = PolynomialRule(n_presyn_neur, n_postsyn_neur)

    @pytest.mark.slow
    def test_oja_net(self):
        """Test Oja Network."""
        ojanet = OjaNet(
            self.poly_rule,
            n_presyn_neur=self.n_presyn_neur,
            n_postsyn_neur=self.n_postsyn_neur,
        )
        presyn_act = torch.randn(2, 1, self.n_presyn_neur)
        postsyn_act = ojanet.forward(presyn_act)

        self.assertEqual(
            list(postsyn_act.shape), [2, 200, self.n_postsyn_neur]
        ) and self.assertEqual(
            type(postsyn_act.grad_fn), type(None)
        ) and self.assertEqual(
            list(ojanet.synaptic_weight.shape),
            [self.n_presyn_neur, self.n_postsyn_neur],
        )
