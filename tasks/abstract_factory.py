"""Base class for task pipeline."""
from abc import ABC, abstractmethod
from argparse import Namespace as NS
from os.path import join
from typing import Callable, Iterable, Optional
import yaml
from syngan.models import AbstractRule, AbstractNeuralNet, Discriminator
import syngan.optimize as optimize
import syngan.utils as utils
import torch


def _resume_from_checkpoint(net, resume_dir, resume_str):
    chpt = torch.load(join(resume_dir, "chpt_models.pt"))
    net.load_state_dict(chpt["%s_state_dict" % resume_str])
    return net


def _get_last_epoch(resume_dir):
    chpt = torch.load(join(resume_dir, "chpt_models.pt"))
    return chpt["epoch"]


class AbstractFactory(ABC):
    """Factory for networks, update rules and optimizer."""

    def __init__(
            self,
            path_to_fit_conf: str,
            path_to_sim_conf: str,
            resume: Optional[bool] = False,
            resume_dir: Optional[str] = None) -> None:
        """
        Set up networks and update rule.

        Args:
            path_to_fit_conf: path to fit configuration.
            path_to_sim_conf: path to simulation / groundtruth configuration.
            resume: if True, resume training from checkpoint
            resume_dir: path to chechpoint files
        """
        super(AbstractFactory, self).__init__()
        self.resume = resume
        self.resume_dir = resume_dir
        self.path_to_sconf = path_to_sim_conf

        if self.resume:
            assert self.resume_dir is not None

        with open(join(path_to_fit_conf, "config.yaml"), "r") as f:
            self.fconf = NS(**yaml.load(f, Loader=yaml.Loader))

        with open(join(path_to_sim_conf, "config.yaml"), "r") as f:
            self.sconf = NS(**yaml.load(f, Loader=yaml.Loader))
        self._check_consistent_hyperparams()

    @abstractmethod
    def make_update_rule(self) -> AbstractRule:
        """Return parametrised update rule."""
        raise NotImplementedError

    @abstractmethod
    def make_neural_network(self) -> AbstractNeuralNet:
        """Return neural network with parametrised update rule."""
        raise NotImplementedError

    @abstractmethod
    def make_discriminator(self) -> Discriminator:
        """Return discriminator network."""
        raise NotImplementedError

    @abstractmethod
    def make_dataloader(self) -> Iterable:
        """Return dataloader."""
        raise NotImplementedError

    def preprocess_in(self, tensor):
        """Preprocessing function for input."""
        return tensor

    def preprocess_targ(self, tensor):
        """Preprocessing function for target."""
        return tensor

    def make_optimizer(
            self,
            logger: Callable) -> optimize.Base:
        """
        Make optimiser.

        Args:
            logger: wandb logger
        """
        # Check for correct loss function
        print("Making networks")
        network = self.make_neural_network()
        network.cuda()

        if network.grad_est != "supervised":
            discriminator = self.make_discriminator()
            discriminator.cuda()

        print("Making dataloader")
        dataloader = self.make_dataloader()

        print("Making optimizer")
        # Check optimizer is implemented
        self._check_loss_funcn()
        assert hasattr(optimize, self.fconf.optimizer_class)

        if "Supervised" in self.fconf.optimizer_class:
            opt =  getattr(optimize, self.fconf.optimizer_class)(
                network=network,
                dataloader=dataloader,
                optim_args=self.fconf.gen_optim_args,
                training_opts=self.fconf.__dict__,
                preprocess_stim=self.preprocess_in,
                preprocess_resp=self.preprocess_targ,
                logger=logger
                )

        else:
            opt = getattr(optimize, self.fconf.optimizer_class)(
                generator=network,
                discriminator=discriminator,
                optim_args=[
                    self.fconf.gen_optim_args,
                    self.fconf.dis_optim_args],
                dataloader=dataloader,
                training_opts=self.fconf.__dict__,
                loss_str=self.fconf.loss_str,
                preprocess_stim=self.preprocess_in,
                preprocess_resp=self.preprocess_targ,
                logger=logger
                )
        if self.resume:
            opt.epoch = _get_last_epoch(self.resume_dir)
        return opt

    @abstractmethod
    def post_processing(self, opt: Callable) -> None:
        """
        Process trained models.

        Args:
            opt: optimizer object
        """
        raise NotImplementedError

    def _check_loss_funcn(self):
        # Workaround for loss functions not in spikegan's loss_dict
        if self.fconf.loss_str not in utils.loss_dict.keys():
            loss_funcn = self.fconf.__dict__.pop("loss_str")
            setattr(self.fconf, "loss_str", "cross_entropy")
            setattr(self.fconf, "new_loss_str", loss_funcn)

    def _check_consistent_hyperparams(self):
        if self.fconf.dis_type == "supervised":
            assert("Supervised" in self.fconf.optimizer_class)
        else:
            assert "Supervised" not in self.fconf.optimizer_class
