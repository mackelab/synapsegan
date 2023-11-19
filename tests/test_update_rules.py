import unittest

import torch
import torch.nn as nn

from syngan.models.update_rules import MLP, OjaMLP, PolynomialRule


class _Reshape(nn.Module):
    def __init__(self, shape) -> None:
        super(_Reshape, self).__init__()
        self.shape = shape

    def forward(self, tensor):
        return tensor.reshape(self.shape)


class TestUpdateRule(unittest.TestCase):
    """Test setup for update rules."""

    n_presyn_neur = 3
    n_postsyn_neur = 2
    degree = 1

    def test_polynomial_expansion(self):
        """Test polynomial expansion rule."""
        poly_rule = PolynomialRule(self.n_presyn_neur, self.n_postsyn_neur, self.degree)
        weight = torch.randn(2, self.n_presyn_neur, self.n_postsyn_neur)
        presyn_act = torch.randn(2, self.n_presyn_neur)
        postsyn_act = torch.randn(2, self.n_postsyn_neur)
        update_weight = poly_rule.forward(weight, presyn_act, postsyn_act).squeeze()
        self.assertEqual(
            list(update_weight.shape), [self.n_presyn_neur, self.n_postsyn_neur]
        )

    def test_mlp(self):
        """Test MLP rule."""
        layers = [
            nn.Linear(11, 6),
            nn.ReLU(),
            _Reshape(shape=(-1, self.n_presyn_neur, self.n_postsyn_neur)),
        ]
        mlp_rule = MLP(
            layers,
            self.n_presyn_neur,
            self.n_postsyn_neur,
        )
        weight = torch.randn(1, self.n_presyn_neur, self.n_postsyn_neur)
        presyn_act = torch.randn(2, self.n_presyn_neur)
        postsyn_act = torch.randn(2, self.n_postsyn_neur)
        update_weight = mlp_rule.forward(weight, presyn_act, postsyn_act).squeeze()
        self.assertEqual(
            list(update_weight.shape), [self.n_presyn_neur, self.n_postsyn_neur]
        )

    def test_ojamlp(self):
        """Test MLP plus Oja rule."""
        layers = [
            nn.Linear(11, 6),
            nn.ReLU(),
            _Reshape(shape=(-1, self.n_presyn_neur, self.n_postsyn_neur)),
        ]
        mlp_rule = OjaMLP(
            layers,
            self.n_presyn_neur,
            self.n_postsyn_neur,
        )
        weight = torch.randn(1, self.n_presyn_neur, self.n_postsyn_neur)
        presyn_act = torch.randn(2, self.n_presyn_neur)
        postsyn_act = torch.randn(2, self.n_postsyn_neur)
        update_weight = mlp_rule.forward(weight, presyn_act, postsyn_act).squeeze()
        self.assertEqual(
            list(update_weight.shape), [self.n_presyn_neur, self.n_postsyn_neur]
        )
