from abc import abstractmethod
from os.path import join
from time import time
from argparse import Namespace as NS
from typing import Callable, List, Optional

import pandas
import numpy as np
import torch
from torch.utils.data import dataloader
from torch.nn.utils import clip_grad_norm_
from syngan.utils import loss_dict


class Base:
    """Base class for all optimisers."""

    def __init__(
        self,
        training_opts: dict,
        logger_metrics: List[str],
        device: torch.device,
        dataloader: dataloader.DataLoader,
        loss_str: Optional[str] = "cross_entropy",
        preprocess_stim: Optional[Callable] = None,
        preprocess_resp: Optional[Callable] = None,
        logger: Optional[Callable] = None,
        grad_est: Optional[str] = None,
    ):
        """
        Set up base class.

        Args:
            training_opts: training hyperparameters (see docs of child classes
                           for more information.)
            logger_metrics: list of metrics to log during training.
            device: CPU or CUDA for training.
            dataloader: iterable dataloader for training and test data.
            loss_str: loss to use for GAN optimisation. See .utils.loss_dict
                      keys for options.
            preprocess_stim: function to preprocess stimulus data from
                             dataloader.
            preprocess_resp: function to preprocess neural responses from
                             dataloader.
            logger: wandb logger. If None, no data is logged.
            grad_est: gradient estimation method
        """
        # Set training options
        self.t_opts = NS(**training_opts)
        assert loss_str in loss_dict.keys()
        self.loss = loss_dict[loss_str]
        self.device = device
        self.epoch = 0
        self.grad_est = grad_est

        # Set up data pre-processing
        self.preprocess_stim = preprocess_stim
        self.preprocess_resp = preprocess_resp
        if self.preprocess_stim is None:
            self.preprocess_stim = lambda x: x
        if self.preprocess_resp is None:
            self.preprocess_resp = lambda x: x

        # Set up train and test data
        self.dataloader = dataloader
        max_hold_out = self.dataloader.dataset.hold_out
        if self.grad_est == "rebar":
            max_hold_out = min(500, max_hold_out)

        if hasattr(self.dataloader.dataset, "inputs_test"):
            self.stim_test, self.resp_test = self.dataloader.dataset.inputs_test
            self.stim_test = self.stim_test[:max_hold_out]
            self.resp_test = self.resp_test[:max_hold_out]
            self.stim_test = self.preprocess_stim(self.stim_test).to(self.device)
            self.resp_test = self.preprocess_resp(self.resp_test).to(self.device)

        # Set up logger and metrics
        self.logger = logger
        self.logger_metrics = logger_metrics
        self.df = pandas.DataFrame(columns=logger_metrics)

    @abstractmethod
    def train(self, *args):
        """Train generative model."""
        raise NotImplementedError

    @abstractmethod
    def _make_checkpoint(self, init=False):
        raise NotImplementedError

    @abstractmethod
    def _log_metrics(self):
        raise NotImplementedError

    @abstractmethod
    def _stop_training(self):
        raise NotImplementedError


class Supervised(Base):
    """Base class for supervised learning."""

    def __init__(
        self,
        network: Callable,
        optim_args: List,
        dataloader: dataloader.DataLoader,
        training_opts: Optional[dict] = dict(max_norm_net=np.inf, track_params=False),
        preprocess_stim: Optional[Callable] = None,
        preprocess_resp: Optional[Callable] = None,
        logger: Optional[Callable] = None,
    ):
        """
        Set up optimiser for supervised learning.

        Args:
            network: neural network to be trained.
            optim_args: list of arguments for the Adam optimiser in the format
                        [learning rate, (beta1, beta2)].
            dataloader: iterable dataloader for training and test data.
            training_opts: training hyperparameters:
                max_norm_net: threshold gradient norm for clipping network
                              gradients.
                track_params: if True, log network parameters.
            preprocess_stim: function to preprocess stimulus data from
                             dataloader.
            preprocess_resp: function to preprocess neural responses from
                             dataloader.
            logger: wandb logger. If None, no data is logged.
        """
        logger_metrics = ["loss", "grad", "global_step"]
        if training_opts["track_params"]:
            for i, _ in enumerate(network.parameters()):
                logger_metrics.append("param_%d" % (i + 1))
        device = list(network.parameters())[0].device

        super(Supervised, self).__init__(
            training_opts,
            logger_metrics,
            device,
            dataloader,
            preprocess_stim=preprocess_stim,
            preprocess_resp=preprocess_resp,
            logger=logger,
            grad_est=None,
        )
        # Make network
        self.network = network
        self.optim_args = optim_args

        # Check if input network is of correct type
        assert self.network.grad_est == "supervised"

        # Make optimiser
        self.optimiser = torch.optim.Adam(self.network.parameters(), *self.optim_args)

        # Checkpoint
        if self.logger is not None:
            self._make_checkpoint(init=True)

    def train(self, epochs: int, log_freq: Optional[int] = 100):
        """
        Train network.

        Args:
            epochs: number of training epochs
            log_freq: epoch frequency for making checkpoints.
        """
        stop_training = False
        epoch = 0

        while not stop_training and epoch < epochs:
            epoch += 1
            self.epoch += 1

            for _, (stim, resp) in self.dataloader:
                self.network.train()
                stim = self.preprocess_stim(stim).to(self.device)
                resp = self.preprocess_resp(resp).to(self.device)

                self._update_network(stim, resp)

            torch.cuda.empty_cache()

            if self.epoch % log_freq == 0:
                if self.logger is not None:
                    print("Logging data.")
                    self._make_checkpoint()
                    if hasattr(self, "stim_test"):
                        self._log_metrics()
                        stop_training = self._stop_training()

    def _net_fwd_pass(self, stim):
        return self.network._hidden_layers[:-1](stim)

    def _update_network(self, stim, resp):
        # Zero gradients
        self.optimiser.zero_grad()

        # Forward pass to get Bernoulli rate
        rate = self._net_fwd_pass(stim)

        # Calculate negative log likelihood and backprop
        pdf = self.network._hidden_layers[-1].pdf(rate, resp)
        loss = -torch.log(pdf).mean()
        loss.backward()

        # Clip gradients
        if self.t_opts.max_norm_net < np.inf:
            clip_grad_norm_(
                self.network.parameters(), max_norm=self.t_opts.max_norm_net
            )

        # Update gradients
        self.optimiser.step()

    def _make_checkpoint(self, init=False):
        checkpoint = {
            "epoch": self.epoch,
            "network_state_dict": self.network.state_dict(),
        }
        init_str = ""
        if init:
            init_str = "_init"
        torch.save(checkpoint, join(self.logger.dir, "chpt_models%s.pt" % init_str))
        torch.save(
            self.dataloader, join(self.logger.dir, "chpt_dataloader%s.pt" % init_str)
        )
        self.df.to_csv(join(self.logger.dir, "logged_metrics.csv"))

    def _log_metrics(self):
        self.network.eval()
        rate_cv = self._net_fwd_pass(self.stim_test)
        pdf_cv = self.network._hidden_layers[-1].pdf(rate_cv, self.resp_test)
        loss_cv = -torch.log(pdf_cv).mean()

        loss_cv.backward()
        net_grad = torch.sqrt(
            sum(
                [
                    torch.norm(p.grad) ** 2
                    for p in self.network.parameters()
                    if p.requires_grad
                ]
            )
        )
        _update_dict = {
            "loss": loss_cv.item(),
            "grad": net_grad.item(),
            "global_step": self.epoch,
        }
        if self.t_opts.track_params:
            for i, p in enumerate(self.network.parameters()):
                _update_dict["param_%d" % (i + 1)] = p.data.cpu().numpy().tolist()
        self.df.loc[self.logger.step] = _update_dict
        torch.cuda.empty_cache()

        # Update logger
        self.logger.history.add(dict(self.df.loc[self.logger.step]))

    def _stop_training(self):
        loss = np.array(self.df.loc[:, ["loss"]][-20:])
        grad = np.array(self.df.loc[:, ["grad"]][-20:])

        if self.epoch > 2000:
            if np.abs(loss).mean() < 1e-6:
                return True
            if np.abs(grad).mean() < 1e-6:
                return True
        if not np.all(
            [
                torch.isfinite(p).all().data.cpu().numpy()
                for p in self.network.parameters()
            ]
        ):
            return True


class Adversarial(Base):
    """Base class for GAN training."""

    def __init__(
        self,
        generator: Callable,
        discriminator: Callable,
        optim_args: List,
        dataloader: dataloader.DataLoader,
        training_opts: Optional[dict] = dict(
            gen_iter=1,
            dis_iter=1,
            max_norm_gen=np.inf,
            max_norm_dis=np.inf,
            track_params=False,
        ),
        loss_str: Optional[str] = "cross_entropy",
        preprocess_stim: Optional[Callable] = None,
        preprocess_resp: Optional[Callable] = None,
        logger: Optional[Callable] = None,
    ):
        """
        Set up optimiser for supervised learning.

        Args:
            generator: GAN generator.
            discriminator: GAN discriminator.
            optim_args: list of arguments for the Adam optimiser in the format
                        [learning rate, (beta1, beta2)].
            dataloader: iterable dataloader for training and test data.
            training_opts: training hyperparameters:
                gen_iter: number of generator updates per epoch.
                dis_iter: number of discriminator updates per epoch.
                max_norm_gen: threshold gradient norm for clipping generator
                              gradients.
                max_norm_dis: threshold gradient norm for clipping
                              discriminator gradients.
                track_params: if True, log parameters of generator.
            loss_str: loss to use for GAN optimisation. See .utils.loss_dict
                      keys for options.
            preprocess_stim: function to preprocess stimulus data from
                             dataloader.
            preprocess_resp: function to preprocess neural responses from
                             dataloader.
            logger: wandb logger. If None, no data is logged.
        """
        logger_metrics = [
            "dis_loss",
            "gen_loss",
            "dis_grad",
            "gen_grad",
            "dreal_mean",
            "dfake_mean",
            "dreal_std",
            "dfake_std",
            "global_step",
        ]
        if training_opts["track_params"]:
            for i, _ in enumerate(generator.parameters()):
                logger_metrics.append("param_%d" % (i + 1))
        if training_opts["track_output"]:
            logger_metrics.append("gen_output")
        if generator.grad_est == "rebar":
            logger_metrics.append("log_temp")
            logger_metrics.append("error_rate")

        device = list(generator.parameters())[0].device

        super(Adversarial, self).__init__(
            training_opts,
            logger_metrics,
            device,
            dataloader,
            loss_str,
            preprocess_stim,
            preprocess_resp,
            logger,
            grad_est=generator.grad_est,
        )
        # Make networks
        self.generator = generator
        self.discriminator = discriminator
        self.optim_args = optim_args

        # Make optimisers
        self.gen_optimiser = torch.optim.Adam(
            self.generator.parameters(), *self.optim_args[0]
        )
        self.dis_optimiser = torch.optim.Adam(
            self.discriminator.parameters(), *self.optim_args[1]
        )

        # Make learning rate annealer
        if hasattr(self.t_opts, "anneal_gen_lr") and (
            self.t_opts.anneal_gen_lr is True
        ):
            assert hasattr(self.t_opts, "anneal_gen_lr_params")
            self.gen_annealer = torch.optim.lr_scheduler.StepLR(
                self.gen_optimiser, **self.t_opts.anneal_gen_lr_params
            )
        if hasattr(self.t_opts, "anneal_dis_lr") and (
            self.t_opts.anneal_dis_lr is True
        ):
            assert hasattr(self.t_opts, "anneal_dis_lr_params")
            self.dis_annealer = torch.optim.lr_scheduler.StepLR(
                self.dis_optimiser, **self.t_opts.anneal_dis_lr_params
            )

        # Checkpoint
        if self.logger is not None:
            self._make_checkpoint(init=True)

    def train(self, epochs: int, log_freq: Optional[int] = 100):
        """
        Train network.

        Args:
            epochs: number of training epochs
            log_freq: epoch frequency for making checkpoints.
        """
        stop_training = False
        epoch = 0

        while not stop_training and epoch < epochs:
            # print("Epoch ", epoch)
            epoch += 1
            self.epoch += 1

            self.discriminator.train()
            self.generator.train()
            tic = time()
            for i, (_, (stim, resp)) in enumerate(self.dataloader):
                stim = self.preprocess_stim(stim).to(self.device)
                resp = self.preprocess_resp(resp).to(self.device)
                self._update_discriminator(stim, resp)

                if i >= self.t_opts.dis_iter:
                    break

            torch.cuda.empty_cache()
            print("Discriminator update", time() - tic)

            tic = time()
            for i, (_, (stim, _)) in enumerate(self.dataloader):
                stim = self.preprocess_stim(stim).to(self.device)
                self._update_generator(stim)
                if i >= self.t_opts.gen_iter:
                    break
            torch.cuda.empty_cache()
            print("Generator update", time() - tic)

            # Update learning rates
            if hasattr(self, "gen_annealer"):
                self.gen_annealer.step()
            if hasattr(self, "dis_annealer"):
                self.dis_annealer.step()

            if self.epoch % log_freq == 0:
                if self.logger is not None:
                    print("Logging data.")
                    self._make_checkpoint()
                    if hasattr(self, "stim_test"):
                        self._log_metrics()
                        stop_training = self._stop_training()

    def _update_generator(self, stim, cv=False):
        # Zero gradients
        self.generator.zero_grad()

        # Forward pass through generator
        resp = self._gen_fwd_pass(stim)

        # Calculate loss
        if self.discriminator.conditional:
            dis_out = self.discriminator([resp, stim])
        else:
            dis_out = self.discriminator(resp)
        loss = self.loss(dis_out).mean()

        # Loss backprop through generator layers
        loss.backward(retain_graph=True)

        # Clip gradients
        norm = None
        if self.t_opts.max_norm_gen < np.inf:
            norm = clip_grad_norm_(
                self.generator.parameters(), max_norm=self.t_opts.max_norm_gen
            )

        # Update generator
        if not cv:
            self.gen_optimiser.step()
        else:
            if norm is None:
                norm = self._compute_grad_norm(self.generator)
            return loss, norm

    def _gen_fwd_pass(self, stim):
        return self.generator(stim)

    def _compute_grad_norm(self, net):
        norm = sum(
            [torch.sum(p.grad ** 2) for p in net.parameters() if p.requires_grad]
        )
        norm = torch.sqrt(norm)
        return norm

    def _update_discriminator(self, stim, resp):
        # Zero gradients
        self.dis_optimiser.zero_grad()

        # Forward through generator
        resp_fake = self._gen_fwd_pass(stim).detach()

        if self.discriminator.conditional:
            d_fake = self.discriminator([resp_fake, stim])
            d_real = self.discriminator([resp, stim])
        else:
            d_fake = self.discriminator(resp_fake)
            print(d_fake.shape)
            d_real = self.discriminator(resp)

        # Calculate loss and backward

        loss = self.loss(d_fake, d_real).mean()
        print(loss)
        loss.backward()

        # Clip gradients
        if self.t_opts.max_norm_dis < np.inf:
            clip_grad_norm_(
                self.discriminator.parameters(), max_norm=self.t_opts.max_norm_dis
            )

        # Update gradients
        self.dis_optimiser.step()

    def _make_checkpoint(self, init=False):
        checkpoint = {
            "epoch": self.epoch,
            "gen_state_dict": self.generator.state_dict(),
            "dis_state_dict": self.discriminator.state_dict(),
        }
        init_str = ""
        if init:
            init_str = "_init"
        torch.save(checkpoint, join(self.logger.dir, "chpt_models%s.pt" % init_str))
        torch.save(
            self.dataloader, join(self.logger.dir, "chpt_dataloader%s.pt" % init_str)
        )
        self.df.to_csv(join(self.logger.dir, "logged_metrics.csv"))

    def _make_update_dict(self):
        # self.generator.eval()
        # self.discriminator.eval()

        resp_fake_cv = self._gen_fwd_pass(self.stim_test).detach()
        if self.discriminator.conditional:
            d_fake_cv = self.discriminator([resp_fake_cv, self.stim_test])
            d_real_cv = self.discriminator([self.resp_test, self.stim_test])
        else:
            d_fake_cv = self.discriminator(resp_fake_cv)
            d_real_cv = self.discriminator(self.resp_test)

        dis_loss_cv = self.loss(d_fake_cv, d_real_cv).mean()
        dis_loss_cv.backward()
        dis_grad = self._compute_grad_norm(self.discriminator)

        gen_loss_cv, gen_grad = self._update_generator(self.stim_test, cv=True)

        _update_dict = {
            "dis_loss": dis_loss_cv.item(),
            "gen_loss": gen_loss_cv.item(),
            "dis_grad": dis_grad.item(),
            "gen_grad": torch.tensor([gen_grad]).squeeze().item(),
            "dreal_mean": d_real_cv.mean().item(),
            "dfake_mean": d_fake_cv.mean().item(),
            "dreal_std": d_real_cv.std().item(),
            "dfake_std": d_fake_cv.std().item(),
            "global_step": self.epoch,
        }
        if self.t_opts.track_params:
            for i, p in enumerate(self.generator.parameters()):
                _update_dict["param_%d" % (i + 1)] = p.data.cpu().numpy().tolist()
        if self.t_opts.track_output:
            _update_dict["gen_output"] = resp_fake_cv.data.cpu().numpy().tolist()

        if self.grad_est == "rebar":
            # TODO: log temp and error rate not logged properly
            _update_dict["log_temp"] = self.log_temp.data.cpu().numpy().squeeze()
            _update_dict["error_rate"] = self.error_rate.data.cpu().numpy().squeeze()
        return _update_dict

    def _log_metrics(self):
        self.df.loc[self.logger.step] = self._make_update_dict()
        torch.cuda.empty_cache()

        # Update logger
        self.logger.log(dict(self.df.loc[self.logger.step]))

    def _stop_training(self):
        lr = self.gen_optimiser.defaults["lr"]
        gen_grad = np.array(self.df.loc[:, ["gen_grad"]][-10:])
        gen_loss = np.array(self.df.loc[:, ["gen_loss"]][-10:])
        d_real_mean = np.array(self.df.loc[:, ["dreal_mean"]][-10:])
        d_fake_mean = np.array(self.df.loc[:, ["dfake_mean"]][-10:])
        d_real_std = np.array(self.df.loc[:, ["dreal_std"]][-10:])
        d_fake_std = np.array(self.df.loc[:, ["dfake_std"]][-10:])

        # If NaNs or Infs in generator params.
        for p in self.generator.parameters():
            if not torch.all(torch.isfinite(p)):
                print("Generator parameters not finite.")
                return True

        # If NaNs or Infs in generator params.
        for p in self.discriminator.parameters():
            if not torch.all(torch.isfinite(p)):
                print("Discriminator parameters not finite.")
                return True

        # If discriminator is overconfident
        if (
            (len(d_fake_std) >= 20)
            and (d_fake_std.mean() < 1e-6)
            and (d_real_std.mean() < 1e-6)
        ):
            print("Discriminator is over-confident.")
            return True

        # If generator loss collapses to 0
        if (len(gen_loss) >= 20) and (np.abs(gen_loss).mean() < 1e-6):
            print("Generator loss is 0.")
            return True

        # If generator gradients are 0.
        if (len(gen_grad) >= 20) and (np.abs(gen_grad.mean()) < 1e-4 * lr):
            print("Generator gradients are 0.")
            return True

        # If discriminator outputs are same for real and fake data.
        if (len(d_real_mean) >= 20) and (
            np.abs(d_real_mean - d_fake_mean).mean() < 1e-3
        ):
            print("Discriminator output is always 0.5")
            return True

        return False
