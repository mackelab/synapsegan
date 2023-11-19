"""Curriculum learning for GAN training."""
import copy
from os.path import join
from time import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from .base import Adversarial
from torch import FloatTensor as FT

import syngan.utils as utils


class CurriculumLearning(Adversarial):
    """Vanilla GAN training with curriculum learning."""

    def __init__(self, **kwargs):
        """Set up GAN training with curriculum learning."""
        super(CurriculumLearning, self).__init__(**kwargs)
        # Add more metrics to track
        if self.t_opts.track_score:
            assert (
                hasattr(self.t_opts, "score_funcn")
                and hasattr(utils, self.t_opts.score_funcn)
                and hasattr(self.t_opts, "path_to_data")
            )
            # TODO: specific to Oja Net. Need to redo for other kinds of
            # networks
            self.score_tr = FT(
                np.load(join(self.t_opts.path_to_data, "training_data.npz"))[
                    "presyn_pcs"
                ]
            ).to(self.device)
            self.logger_metrics.append("score")
        if self.t_opts.track_weight:
            self.logger_metrics.append("weight")
        self.df = pd.DataFrame(columns=self.logger_metrics)

        assert hasattr(self.t_opts, "curriculum")
        self.curriculum = self.t_opts.curriculum
        self.current = self.curriculum[0][1]
        if hasattr(self, "resp_test"):
            self.resp_test_tot = copy.deepcopy(self.resp_test)
        if self.t_opts.no_cv:
            stim_test, resp_test = self.dataloader.dataset.inputs
            self.stim_test = self.preprocess_stim(stim_test).to(self.device)
            self.resp_test_tot = resp_test.to(self.device)
            self.resp_test = copy.deepcopy(self.resp_test_tot)

    def _gen_fwd_pass(self, stim, **kwargs):
        return self.generator(stim, timesteps=self.current, **kwargs)

    def train(self, epochs: int, log_freq: Optional[int] = 100):
        """
        Train network.

        Args:
            epochs: number of training epochs
            log_freq: epoch frequency for making checkpoints.
        """
        stop_training = False
        epoch = 0
        curr_epochs = np.cumsum([curr[0] for curr in self.curriculum])
        # curr_tsteps = [curr[1] for curr in self.curriculum]

        while not stop_training and epoch < epochs:
            # print("Epoch ", epoch)
            epoch += 1
            self.epoch += 1

            # Get current curriculum
            inds = np.argwhere(curr_epochs / self.epoch >= 1)
            if len(inds) > 0:
                ind = inds[0].squeeze()
            else:
                ind = -1
            self.current = self.curriculum[ind][1]
            print(self.epoch, ind, self.current)

            if hasattr(self, "resp_test"):
                self.resp_test = self.resp_test_tot[..., : self.current, :]
                self.resp_test = self.preprocess_resp(self.resp_test).to(self.device)

            self.discriminator.train()
            self.generator.train()
            tic = time()
            for i, (_, (stim, resp)) in enumerate(self.dataloader):
                resp = resp[..., : self.current, :]
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

            if self.epoch % log_freq == 0:
                if self.logger is not None:
                    print("Logging data.")
                    self._make_checkpoint()
                    if hasattr(self, "stim_test"):
                        self._log_metrics()
                        stop_training = self._stop_training()

    def _make_update_dict(self):
        _update_dict = super(CurriculumLearning, self)._make_update_dict()
        weights = None
        if self.t_opts.track_weight:
            weights, _ = self._gen_fwd_pass(self.stim_test, return_weight=True)
            _update_dict["weight"] = weights.data.cpu().numpy().tolist()

        # TODO: specific to Oja Net. Need to redo for other kinds of networks
        if self.t_opts.track_score:
            if weights is None:
                weights, _ = self._gen_fwd_pass(self.stim_test, return_weight=True)
            score_gt = self.score_tr[-self.t_opts.hold_out:].to(self.device)

            score = [
                getattr(utils, self.t_opts.score_funcn)(
                    score_gt,
                    wt.to(self.device)
                    if wt is not None
                    else self.generator.current_weight,
                )
                .data.cpu()
                .numpy()
                .tolist()
                for wt in weights
            ]
            _update_dict["score"] = score
        return _update_dict
