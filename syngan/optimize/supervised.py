import numpy as np
import pandas as pd
import torch
from .base import Supervised
from torch.nn.utils import clip_grad_norm_

import syngan.utils as utils


# TODO: applies ONLY to OjaNet
class OjaSupervised(Supervised):
    """Supevised training with Oja Net loss."""

    def __init__(self, **kwargs):
        """Set up optimiser for supervised learning."""
        super(OjaSupervised, self).__init__(**kwargs)
        assert hasattr(utils, self.t_opts.new_loss_str)

        if self.t_opts.track_score:
            assert hasattr(self.t_opts, "score_funcn") and hasattr(
                utils, self.t_opts.score_funcn
            )
            self.logger_metrics.append("score")
        if self.t_opts.track_weight:
            self.logger_metrics.append("weight")
        self.df = pd.DataFrame(columns=self.logger_metrics)

        if self.t_opts.no_cv:
            stim_test, resp_test = self.dataloader.dataset.inputs
            self.stim_test = self.preprocess_stim(stim_test).to(self.device)
            self.resp_test = resp_test.to(self.device)

    def _net_fwd_pass(self, stim):
        return self.network.forward(stim, return_weight=True)

    def _update_network(self, stim, resp):
        # Zero gradients
        self.optimiser.zero_grad()

        weights, activity = self._net_fwd_pass(stim)
        if self.t_opts.new_loss_str == "pc_norm":
            inputs = weights[-1]
        else:
            inputs = activity

        loss = getattr(utils, self.t_opts.new_loss_str)(resp, inputs)
        loss.backward()

        # Clip gradients
        if self.t_opts.max_norm_net < np.inf:
            clip_grad_norm_(
                self.network.parameters(), max_norm=self.t_opts.max_norm_net
            )

        # Update step
        self.optimiser.step()

    def _log_metrics(self):
        self.network.eval()
        weights, activity = self._net_fwd_pass(self.stim_test)
        if self.t_opts.new_loss_str == "pc_norm":
            loss_cv = getattr(utils, self.t_opts.new_loss_str)(
                self.resp_test,
                weights[-1])
        else:
            loss_cv = getattr(utils, self.t_opts.new_loss_str)(
                self.resp_test,
                activity)
        loss_cv.backward()
        net_grad = torch.sqrt(
            sum([torch.norm(p.grad) ** 2 for p in self.network.parameters()])
        )
        _update_dict = {
            "loss": loss_cv.item(),
            "grad": net_grad.item(),
            "global_step": self.epoch,
        }
        if self.t_opts.track_params:
            for i, p in enumerate(self.network.parameters()):
                _update_dict["param_%d" % (i + 1)] = p.data.cpu().numpy().tolist()

        if self.t_opts.track_weight:
            _update_dict["weight"] = weights.data.cpu().numpy().tolist()
        if self.t_opts.track_score:
            score = [
                getattr(utils, self.t_opts.score_funcn)(
                    self.resp_test,
                    wt if wt is not None else self.network.current_weight,
                )
                .data.cpu()
                .numpy()
                .tolist()
                for wt in weights
            ]
            _update_dict["score"] = score

        self.df.loc[self.logger.step] = _update_dict
        torch.cuda.empty_cache()

        # Update logger
        self.logger.log(dict(self.df.loc[self.logger.step]))
