"""Pipeline for experiments with Oja's Rule and bigger neural network."""
from tasks.oja_net_small import Factory as AF
from tasks.abstract_factory import _resume_from_checkpoint
import syngan.models.update_rules as update_rules
from syngan.models import (
    AbstractRule, AbstractNeuralNet, Discriminator, ModuleWrapper, OjaNet, Collapse)
from syngan.utils import make_dataloader as util_make_dataloader
import numpy as np
import torch
from torch import nn, FloatTensor as FT
import torch.nn.utils.spectral_norm as sn
from typing import Iterable, Callable
from os.path import join


class Factory(AF):
    def __init__(
            self,
            path_to_fit_conf="./tasks/oja_net_big/",
            path_to_sim_conf="../data/oja_net_big/",
            resume=False,
            resume_dir=None) -> None:
        """Set up factory for Oja's rule and big network pipeline."""
        super(Factory, self).__init__(
            path_to_fit_conf, path_to_sim_conf, resume, resume_dir)

    def make_update_rule(self) -> AbstractRule:
        """Make update rule."""
        assert hasattr(update_rules, self.fconf.rule_class)
        if "MLP" in self.fconf.rule_class:
            if "Local" in self.fconf.rule_class:
                layers = [nn.Linear(3, 100, bias=False),
                          nn.Sigmoid(),
                          nn.Linear(100, 100, bias=False),
                          nn.Sigmoid(),
                          nn.Linear(100, 1, bias=False),
                          ModuleWrapper(torch.reshape,
                                        shape=(-1,
                                               self.sconf.n_presyn_neur,
                                               self.sconf.n_postsyn_neur))
                          ]
            elif "Aggregate" in self.fconf.rule_class:
                layers = [nn.Linear(5, 100, bias=False),
                          nn.Sigmoid(),
                          nn.Linear(100, 100, bias=False),
                          nn.Sigmoid(),
                          nn.Linear(100, 1, bias=False),
                          ModuleWrapper(torch.reshape,
                                        shape=(-1,
                                               self.sconf.n_presyn_neur,
                                               self.sconf.n_postsyn_neur))
                          ]
            else:
                num_feat = self.sconf.n_presyn_neur * \
                    self.sconf.n_postsyn_neur + \
                    self.sconf.n_presyn_neur + self.sconf.n_postsyn_neur

                layers = [nn.Linear(num_feat, 100, bias=False),
                          nn.Sigmoid(),
                          nn.Linear(100, 100, bias=False),
                          nn.Sigmoid(),
                          nn.Linear(100,
                                    self.sconf.n_postsyn_neur *
                                    self.sconf.n_presyn_neur,
                                    bias=False),
                          ModuleWrapper(torch.reshape,
                                        shape=(-1,
                                               self.sconf.n_presyn_neur,
                                               self.sconf.n_postsyn_neur))
                          ]

            network = getattr(update_rules, self.fconf.rule_class)(
                layers=layers,
                **self.fconf.rule_class_args)

        else:
            network = getattr(
                update_rules,
                self.fconf.rule_class)(**self.fconf.rule_class_args)

        return network

    def make_neural_network(self) -> AbstractNeuralNet:
        """Make rate network."""
        rule = self.make_update_rule()
        network = OjaNet(
            update_rule=rule,
            update_rate=self.fconf.update_rate,
            n_presyn_neur=self.sconf.n_presyn_neur,
            n_postsyn_neur=self.sconf.n_postsyn_neur,
            timesteps=self.fconf.timesteps,
            noise_amplitude=self.sconf.noise_amplitude)

        # Need to do this as a workaround for spikegan set up
        if self.fconf.dis_type == "supervised":
            setattr(network, "grad_est", "supervised")
        else:
            setattr(network, "grad_est", "concrete_relax")

        if self.fconf.init_gen_from_previous:
            chpt = torch.load(self.fconf.path_to_previous_gen)
            network.load_state_dict(chpt["gen_state_dict"])

        # Resume from checkpoint
        if self.resume:
            network = _resume_from_checkpoint(
                network,
                self.resume_dir,
                "network" if network.grad_est == "supervised" else "gen")

        return network

    def make_discriminator(self) -> Discriminator:
        """Make discriminator network."""
        layers = [ModuleWrapper(torch.transpose, dim0=-2, dim1=-1),
                  sn(nn.Conv1d(in_channels=self.sconf.n_postsyn_neur,
                               out_channels=5,
                               kernel_size=10)),
                  nn.LeakyReLU(0.2),
                  sn(nn.Conv1d(in_channels=5,
                               out_channels=1,
                               kernel_size=10)),
                  nn.LeakyReLU(0.2),
                  Collapse(),
                  sn(nn.Linear(182, 100)),
                  nn.LeakyReLU(0.2),
                  sn(nn.Linear(100, 100)),
                  nn.LeakyReLU(0.2),
                  sn(nn.Linear(100, 1)),
                  nn.Sigmoid()
                  ]
        discriminator = Discriminator(layers, conditional=False)

        # Resume from checkpoint
        if self.resume:
            discriminator = _resume_from_checkpoint(discriminator,
                                                    self.resume_dir,
                                                    "dis")
        elif self.fconf.init_dis_from_previous:
            chpt = torch.load(self.fconf.path_to_previous_dis)
            discriminator.load_state_dict(chpt["dis_state_dict"])

        return discriminator

    def make_dataloader(self) -> Iterable:
        """Make dataloader."""
        training_data = np.load(join(self.path_to_sconf,
                                     "training_data.npz"
                                     )
                                )
        X, Y, PCs = FT(training_data["presyn_act"]), \
            FT(training_data["postsyn_act"][..., :self.fconf.timesteps]), \
            FT(training_data["presyn_pcs"])

        if self.fconf.dis_type == "supervised" and self.fconf.loss_str == "pc_norm":
            inputs_list = [X,
                           PCs
                           ]
        else:
            inputs_list = [X,
                           Y.transpose(-1, -2)
                           ]

        inputs = {"_inputs": inputs_list,
                  "hold_out": self.fconf.hold_out}
        return util_make_dataloader(self.fconf.batch_size,
                               inputs_to_dataset_class=inputs,
                               )

    def post_processing(self, opt: Callable) -> None:
        """
        Process trained models.

        Args:
            opt: optimizer object
        """
        test_data = np.load(join(self.path_to_sconf, "test_data.npz"))
        X = test_data["presyn_act"]

        if self.fconf.dis_type == "supervised":
            network = opt.network
        else:
            network = opt.generator
        weights, postsyn_act = [], []

        for x in X:
            w, out = network(FT(x), timesteps=200, return_weight=True)
            weights.append(w)
            postsyn_act.append(out)
        weights = torch.cat(weights, 0).data.cpu().numpy()
        postsyn_act = torch.cat(postsyn_act, 0).data.cpu().numpy()

        np.savez(join(opt.logger.dir, "model_test_predictions.npz"),
                 weights=weights, postsyn_act=postsyn_act)
