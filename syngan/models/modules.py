from typing import Callable, List

import torch
import torch.nn as nn


class ModuleWrapper(nn.Module):
    """Class to wrap a torch function."""

    def __init__(self, func: Callable, **kwargs):
        """
        Wrap a torch function (eg. math operations) as an nn.Module object.

        For example: `ModuleWrapper(torch.exp)` makes torch.exp operation an
        nn.Module object that can then be used as a hidden layer in a torch
        network.

        Args:
            func: torch function
            **kwargs: keyword arguments to `func`
        """
        super(ModuleWrapper, self).__init__()
        self.func = func
        self.func_args = kwargs

    def forward(self, _input: torch.tensor) -> torch.tensor:
        """Forward pass."""
        return self.func(_input, **self.func_args)


class ChooseOutput(nn.Module):
    """Class for choosing from multiple outputs."""

    def __init__(self, ind: int):
        """
        Set up class for choosing outputs.

        Args:
            ind: index of ouotput from previous layer.
        """
        super(ChooseOutput, self).__init__()
        self.ind = ind

    def forward(self, _input: List[torch.tensor]) -> torch.tensor:
        """Forward pass."""
        return _input[self.ind]


class Collapse(nn.Module):
    """Collapse tensor to 2D."""

    def __init__(self):
        """Collapse tensor to 2D."""
        super(Collapse, self).__init__()

    def forward(self, input: torch.tensor) -> torch.tensor:
        """Forward pass."""
        shape = input.shape
        return input.reshape(shape[0], -1)
