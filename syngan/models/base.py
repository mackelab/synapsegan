"""Base classes for update rules and biological neural networks."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class AbstractRule(ABC, nn.Module):
    """Abstract base class for update rules."""

    def __init__(self):
        """Set up abstract base class."""
        super(AbstractRule, self).__init__()
        # all child classes must have a parameter attribute
        self.parameter = None
        self.n_presyn_neurs = 1
        self.n_postsyn_neurs = 1


    # all child classes should implement a forward method with inputs
    # in the specified order
    @abstractmethod
    def forward(
        self,
        weight: torch.tensor,
        presyn_activity: torch.tensor,
        postsyn_act: torch.tensor,
    ) -> torch.tensor:
        """
        Forward pass to update synaptic weights.

        Args:
            weight: old synaptic weights
            presyn_activity: activity of presynaptic neurons.
            postsyn_activity: activity of postsynaptic_neurons.
        """
        raise NotImplementedError

    def _check(self, weight, presyn_activity, postsyn_activity):
        assert list(weight.shape)[-2:] == [self.n_presyn_neur, self.n_postsyn_neur]
        assert presyn_activity.shape[-1] == self.n_presyn_neur
        assert postsyn_activity.shape[-1] == self.n_postsyn_neur


class AbstractNeuralNet(ABC, nn.Module):
    """Abstract base class for rate / spiking networks."""

    def __init__(
        self,
        update_rule: AbstractRule,
        update_rate: Optional[float],
    ):
        r"""
        Set up abstract base class.

        Args:
            update_rule: class containing the parameterised update rule.
            update_rate: learning rate $\eta$ for the update rule.
        """
        super(AbstractNeuralNet, self).__init__()
        self.update_rule = update_rule
        self.update_rate = update_rate
        self.current_weight = None
        self.wt_init = None

    @abstractmethod
    def update_synaptic_weight(self, *args):
        """Update synaptic weights."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, **args) -> torch.tensor:
        """Forward pass to get post-synaptic activity."""
        raise NotImplementedError

    # @abstractmethod
    # def loss_funcn(self, *args):
    #     """Network-specific loss function."""
    #     raise NotImplementedError
