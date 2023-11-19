"""Modules for parameterized update rules."""

from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn

from . import AbstractRule


class OjaRule(AbstractRule):
    """Groundtruth Oja rule."""

    def __init__(self, n_presyn_neur: int, n_postsyn_neur: int):
        """
        Set up groundtruth rule.

        Args:
            n_presyn_neur: number of presynaptic neurons.
            n_postsyn_neur: number of postsynaptic neurons.
        """
        super(OjaRule, self).__init__()
        self.n_presyn_neur = n_presyn_neur
        self.n_postsyn_neur = n_postsyn_neur

        self.parameter = None

    def forward(self, weight, presyn_activity, postsyn_activity):
        """Forward pass."""
        # Chack shapes of input
        self._check(weight, presyn_activity, postsyn_activity)
        wt = weight.reshape(1, self.n_presyn_neur, self.n_postsyn_neur)
        pre = presyn_activity.reshape(len(presyn_activity),
                                      self.n_presyn_neur,
                                      1)
        post = postsyn_activity.reshape(len(presyn_activity),
                                        1,
                                        self.n_postsyn_neur)
        up_wt = ((post * pre) - ((post ** 2) * wt)).mean(0)
        return up_wt.unsqueeze(0)


class PolynomialRule(AbstractRule):
    """Polynomial expansion with parameterised coefficients."""

    def __init__(
        self, n_presyn_neur: int, n_postsyn_neur: int, degree: Optional[int] = 2
    ):
        """
        Set up polynomial expansion rule.

        Args:
            n_presyn_neur: number of presynaptic neurons.
            n_postsyn_neur: number of postsynaptic neurons.
            degree: degree of the polynomial
        """
        super(PolynomialRule, self).__init__()
        self.n_presyn_neur = n_presyn_neur
        self.n_postsyn_neur = n_postsyn_neur
        self.degree = degree
        self.num_features = 3

        # compute number of terms in the polynomial
        self.num_terms = (self.degree + 1) ** (self.num_features)
        self.parameter = nn.Linear(self.num_terms, 1, bias=False)
        self.parameter.weight.data = 0.1 * torch.randn(self.num_terms)

    def _check_params(self):
        if torch.any(torch.sum(self.parameter.weight) > 1.0):
            print("Non finite weights. Resetting to zero.")
            wt = self.parameter.weight.data
            self.parameter.weight.data = wt / (torch.norm(wt) + 1e-3)

    def forward(self, weight, presyn_activity, postsyn_activity):
        """Forward pass."""
        # Chack shapes of input
        self._check(weight, presyn_activity, postsyn_activity)

        self._check_params()
        # Get powers for different features
        ran = range(self.degree + 1)
        x, y, z = np.meshgrid(ran, ran, ran)

        # Get terms for polynomial expansion
        terms = [
            (
                (weight ** a)
                * (presyn_activity.unsqueeze(-1) ** b)
                * (postsyn_activity.unsqueeze(-2) ** c)
            ).unsqueeze(-1)
            for a, b, c in zip(x.flatten(), y.flatten(), z.flatten())
        ]
        terms = torch.cat(terms, -1)
        assert terms.shape[-1] == self.num_terms

        return self.parameter(terms).mean(0).unsqueeze(0)


class MinimalPolynomialRule(AbstractRule):
    """Minimal polynomial expansion with parameterised coefficients."""

    def __init__(self, n_presyn_neur: int, n_postsyn_neur: int):
        """
        Set up minimal polynomial expansion rule.

        Args:
            n_presyn_neur: number of presynaptic neurons.
            n_postsyn_neur: number of postsynaptic neurons.
        """
        super(MinimalPolynomialRule, self).__init__()
        self.n_presyn_neur = n_presyn_neur
        self.n_postsyn_neur = n_postsyn_neur

        # compute number of terms in the polynomial
        self.parameter_1 = nn.Parameter(0.1 * torch.randn(1), requires_grad=True)
        self.parameter_2 = nn.Parameter(0.1 * torch.randn(1), requires_grad=True)

    def forward(self, weight, presyn_activity, postsyn_activity):
        """Forward pass."""
        # Chack shapes of input
        self._check(weight, presyn_activity, postsyn_activity)

        # Get terms for polynomial expansion
        pre = presyn_activity.reshape(len(presyn_activity), self.n_presyn_neur, 1)
        post = postsyn_activity.reshape(len(presyn_activity), 1, self.n_postsyn_neur)
        out = self.parameter_1 * pre * post + self.parameter_2 * (pre ** 2) * weight
        return out.mean(0).unsqueeze(0)


class MLP(AbstractRule):
    """Multi-layer perceptron encoding update rule."""

    def __init__(
        self,
        layers: List[nn.Module],
        n_presyn_neur: int,
        n_postsyn_neur: int,
    ):
        """
        Set up MLP.

        Args:
            layers: list of hidden layers for generator network.
            n_presyn_neur: number of presynaptic neurons.
            n_postsyn_neur: number of postsynaptic neurons.
        """
        super(MLP, self).__init__()
        self.parameter = nn.Sequential(*layers)
        self.n_presyn_neur = n_presyn_neur
        self.n_postsyn_neur = n_postsyn_neur
        self.num_in_feat = (
            (self.n_presyn_neur * self.n_postsyn_neur)
            + self.n_postsyn_neur
            + self.n_presyn_neur
        )
        assert self.parameter[0].in_features == self.num_in_feat

    def forward(self, weight, presyn_activity, postsyn_activity):
        """Forward pass."""
        # Reshape all variables to batch_size x -1
        weight = weight.reshape(1, -1).repeat(len(presyn_activity), 1)
        presyn_activity = presyn_activity.reshape(len(presyn_activity), -1)
        postsyn_activity = postsyn_activity.reshape(len(postsyn_activity), -1)

        # Make input features by concatenating weights and activities
        in_feat = torch.cat([weight, presyn_activity, postsyn_activity], -1)

        return self.parameter(in_feat).mean(0).unsqueeze(0)


class AggregateMLP(AbstractRule):
    """MLP update rule getting aggregate non-local information."""

    def __init__(
        self,
        layers: List[nn.Module],
        n_presyn_neur: int,
        n_postsyn_neur: int,
    ):
        """
        Set up MLP.

        Args:
            layers: list of hidden layers for generator network.
            n_presyn_neur: number of presynaptic neurons.
            n_postsyn_neur: number of postsynaptic neurons.
        """
        super(AggregateMLP, self).__init__()
        self.parameter = nn.Sequential(*layers)
        self.n_presyn_neur = n_presyn_neur
        self.n_postsyn_neur = n_postsyn_neur
        self.num_in_feat = 5
        assert self.parameter[0].in_features == self.num_in_feat

    def forward(self, weight, presyn_activity, postsyn_activity):
        """Forward pass."""
        # Reshape all variables to batch_size x -1
        W = weight.reshape(1, self.n_presyn_neur, self.n_postsyn_neur, 1).repeat(
            len(presyn_activity), 1, 1, 1
        )
        X = presyn_activity.reshape(
            len(presyn_activity), self.n_presyn_neur, 1, 1
        ).repeat(1, 1, self.n_postsyn_neur, 1)
        Y = postsyn_activity.reshape(
            len(postsyn_activity), 1, self.n_postsyn_neur, 1
        ).repeat(1, self.n_presyn_neur, 1, 1)
        agg_X = X.mean(1).unsqueeze(1).repeat(1, self.n_presyn_neur, self.n_postsyn_neur, 1)
        agg_W = W.mean(1).unsqueeze(1).repeat(1, self.n_presyn_neur, self.n_postsyn_neur, 1)

        inputs = torch.cat([X, Y, W, agg_X, agg_W], -1).reshape(-1, 5)

        return self.parameter(inputs
                              ).reshape(
            -1, self.n_presyn_neur, self.n_postsyn_neur
                              ).mean(0).unsqueeze(0)


class LocalMLP(AbstractRule):
    """Multi-layer perceptron encoding a local update rule."""

    def __init__(
        self, layers: List[nn.Module], n_presyn_neur: int, n_postsyn_neur: int
    ):
        """
        Set up local MLP rule.

        Args:
            layers: list of hidden layers for generator network.
            n_presyn_neur: number of presynaptic neurons.
            n_postsyn_neur: number of postsynaptic neurons.
        """
        super(LocalMLP, self).__init__()
        self.parameter = nn.Sequential(*layers)
        self.n_presyn_neur = n_presyn_neur
        self.n_postsyn_neur = n_postsyn_neur
        assert self.parameter[0].in_features == 3

    def forward(self, weight, presyn_activity, postsyn_activity):
        """Forward pass."""
        W = weight.reshape(1, self.n_presyn_neur, self.n_postsyn_neur, 1).repeat(
            len(presyn_activity), 1, 1, 1
        )
        X = presyn_activity.reshape(
            len(presyn_activity), self.n_presyn_neur, 1, 1
        ).repeat(1, 1, self.n_postsyn_neur, 1)
        Y = postsyn_activity.reshape(
            len(postsyn_activity), 1, self.n_postsyn_neur, 1
        ).repeat(1, self.n_presyn_neur, 1, 1)
        inputs = torch.cat([X, Y, W], -1).reshape(-1, 3)
        output = (
            self.parameter(inputs)
            .reshape(-1, self.n_presyn_neur, self.n_postsyn_neur).mean(0).unsqueeze(0)
        )
        return output


class OjaMLP(AbstractRule):
    """Multi-layer perceptron plus Oja Rule for synaptic update rule."""

    def __init__(
        self,
        layers: List[nn.Module],
        n_presyn_neur: int,
        n_postsyn_neur: int,
    ):
        """
        Set up MLP.

        Args:
            layers: list of hidden layers for generator network.
            n_presyn_neur: number of presynaptic neurons.
            n_postsyn_neur: number of postsynaptic neurons.
        """
        super(OjaMLP, self).__init__()
        self.parameter = nn.Sequential(*layers)
        self.n_presyn_neur = n_presyn_neur
        self.n_postsyn_neur = n_postsyn_neur
        self.num_in_feat = (
            (self.n_presyn_neur * self.n_postsyn_neur)
            + self.n_postsyn_neur
            + self.n_presyn_neur
        )
        assert self.parameter[0].in_features == self.num_in_feat

    def forward(self, weight, presyn_activity, postsyn_activity):
        """Forward pass."""
        pre = presyn_activity.reshape(len(presyn_activity), self.n_presyn_neur, 1)
        post = postsyn_activity.reshape(len(presyn_activity), 1, self.n_postsyn_neur)
        up_wt = ((post * pre) - ((post ** 2) * weight)).mean(0)

        weight = weight.reshape(1, -1).repeat(len(presyn_activity), 1)
        presyn_activity = presyn_activity.reshape(len(presyn_activity), -1)
        postsyn_activity = postsyn_activity.reshape(len(postsyn_activity), -1)
        in_feat = torch.cat([weight, presyn_activity, postsyn_activity], -1)

        return self.parameter(in_feat).mean(0).unsqueeze(0) + up_wt


class LocalOjaMLP(AbstractRule):
    """Multi-layer perceptron plus Oja's Rule encoding a local update rule."""

    def __init__(
        self,
        layers: List[nn.Module],
        n_presyn_neur: int,
        n_postsyn_neur: int,
    ):
        """
        Set up local MLP rule.

        Args:
            layers: list of hidden layers for generator network.
            n_presyn_neur: number of presynaptic neurons.
            n_postsyn_neur: number of postsynaptic neurons.
        """
        super(LocalOjaMLP, self).__init__()
        self.parameter = nn.Sequential(*layers)
        self.n_presyn_neur = n_presyn_neur
        self.n_postsyn_neur = n_postsyn_neur
        assert self.parameter[0].in_features == 3

    def forward(self, weight, presyn_activity, postsyn_activity):
        """Forward pass."""
        weight = weight.reshape(1, self.n_presyn_neur, self.n_postsyn_neur, 1).repeat(
            len(presyn_activity), 1, 1, 1
        )
        presyn_activity = presyn_activity.reshape(
            len(presyn_activity), self.n_presyn_neur, 1, 1
        ).repeat(1, 1, self.n_postsyn_neur, 1)
        postsyn_activity = postsyn_activity.reshape(
            len(postsyn_activity), 1, self.n_postsyn_neur, 1
        ).repeat(1, self.n_presyn_neur, 1, 1)
        inputs = torch.cat([presyn_activity, postsyn_activity, weight], -1).reshape(
            -1, 3
        )
        output1 = self.parameter(inputs).mean(0).unsqueeze(0)
        output2 = (
            (postsyn_activity * (presyn_activity - (postsyn_activity * weight)))
            .reshape(-1, self.n_presyn_neur, self.n_postsyn_neur)
            .mean(0).unsqueeze(0)
        )
        return output1 + output2
