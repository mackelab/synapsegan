import unittest
import pytest

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import dataloader

from syngan.models.ojas_net import OjaNet
from syngan.models.update_rules import MinimalPolynomialRule
from syngan.optimize.curriculum_learning import CurriculumLearning
from syngan.optimize.supervised import OjaSupervised


class _Transpose(nn.Module):
    def __init__(self, dim0, dim1) -> None:
        super(_Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, tensor):
        return tensor.transpose(self.dim0, self.dim1)


class _Collapse(nn.Module):
    def __init__(self) -> None:
        super(_Collapse, self).__init__()

    def forward(self, tensor):
        return tensor.reshape(len(tensor), -1)


class _Concat(nn.Module):
    def __init__(self) -> None:
        super(_Concat, self).__init__()

    def forward(self, tensor):
        return torch.cat(tensor, 1)


class _Dataset(data.Dataset):
    def __init__(self) -> None:
        super(_Dataset, self).__init__()
        self.hold_out = 1
        self.inputs = [torch.randn(10, 3), torch.randn(10, 100, 1)]
        self.inputs_test = [torch.randn(1, 3), torch.randn(1, 100, 1)]

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        batch_inputs = []
        for inp in self.inputs:
            batch_inputs.append(inp[idx])
        return idx, batch_inputs


class TestOptimizers(unittest.TestCase):
    """Test setup for optimizers."""

    n_presyn_neur = 3
    n_postsyn_neur = 1
    timesteps = 100
    poly_rule = MinimalPolynomialRule(n_presyn_neur, n_postsyn_neur)
    network = OjaNet(
        update_rule=poly_rule,
        n_presyn_neur=n_presyn_neur,
        n_postsyn_neur=n_postsyn_neur,
        timesteps=timesteps,
    )
    dis = nn.Sequential(
        _Transpose(dim0=-2, dim1=-1),
        nn.Conv1d(in_channels=n_postsyn_neur, out_channels=5, kernel_size=10),
        # nn.LeakyReLU(0.2),
        _Collapse(),
        # nn.Linear(455, 1),
        nn.Sigmoid(),
    )
    setattr(dis, "conditional", False)
    dataloader = dataloader.DataLoader(_Dataset(), 1, shuffle=True)

    @pytest.mark.slow
    def test_curriculum_learning(self):
        """Test curriculum learning optimizer."""
        setattr(self.network, "grad_est", "concrete_relax")
        optimizer = CurriculumLearning(
            generator=self.network,
            discriminator=self.dis,
            optim_args=[(0.0001, (0.9, 0.99)), (0.0001, (0.9, 0.999))],
            dataloader=self.dataloader,
            training_opts={
                "hold_out": 1,
                "batch_size": 1,
                "update_rate": 0.1,
                "gen_iter": 1,
                "dis_iter": 1,
                "max_norm_gen": 0.1,
                "max_norm_dis": 0.1,
                "curriculum": [[2, 100]],
                "track_score": False,
                "track_weight": False,
                "track_params": False,
                "track_output": False,
                "no_cv": False,
            },
            loss_str="cross_entropy",
            preprocess_stim=None,
            preprocess_resp=None,
            logger=None,
        )
        optimizer.train(2)
        self.assertEqual(optimizer.epoch, 2)

    def test_oja_supervised(self):
        """Test supervised learning optimizer for Oja's rule."""
        setattr(self.network, "grad_est", "supervised")
        optimizer = OjaSupervised(
            network=self.network,
            optim_args=(0.0001, (0.9, 0.99)),
            dataloader=self.dataloader,
            training_opts={
                "hold_out": 1,
                "batch_size": 1,
                "update_rate": 0.1,
                "gen_iter": 1,
                "dis_iter": 1,
                "max_norm_net": 0.1,
                "curriculum": [[2, 10]],
                "track_score": False,
                "track_weight": False,
                "track_params": False,
                "track_output": False,
                "no_cv": False,
                "new_loss_str": "pc_norm",
            },
            preprocess_stim=None,
            preprocess_resp=None,
            logger=None,
        )
        optimizer.train(2)
        self.assertEqual(optimizer.epoch, 2)
