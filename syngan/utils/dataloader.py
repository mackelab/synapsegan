from typing import Callable, Iterable, List, Optional

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import dataloader


class MakeDataset(data.Dataset):
    """Make dataset for training and testing."""

    def __init__(self, _inputs: List[torch.tensor], hold_out: Optional[int] = 100):
        """
        Set up class to make dataset.

        Args:
            _inputs: list of inputs that will be passed to dataloader
            hold_out: number of held-out datapoints for testing / validation.
        """
        self.inputs = _inputs
        self.hold_out = hold_out

        # Check all input lengths are same
        assert np.all([len(inp) == len(self.inputs[0]) for inp in self.inputs[1:]])
        # Check that hold out is smaller that length of dataset length
        assert len(self.inputs[0]) > hold_out

        if self.hold_out > 0:
            self.inputs_test = [inp[-self.hold_out:] for inp in self.inputs]

    def __len__(self):
        """Return length of dataset."""
        return len(self.inputs[0]) - self.hold_out

    def __getitem__(self, idx):
        """Get data at input index."""
        batch_inputs = []
        for inp in self.inputs:
            batch_inputs.append(inp[idx])
        return idx, batch_inputs


def make_dataloader(
    batch_size: int,
    inputs_to_dataset_class: dict,
    dataset_class: Optional[Callable] = MakeDataset,
) -> Iterable:
    """
    Make iterable dataloader.

    Args:
        batch_size: batch size.
        inputs_to_dataset_class: dictionary of arguments for Dataset class.
        dataset_class: class to make dataset.
    """
    dataset = dataset_class(**inputs_to_dataset_class)
    return dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=True)
