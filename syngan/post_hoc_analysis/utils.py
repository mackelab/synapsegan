"""Code for loading trained networks."""
import importlib
from argparse import Namespace as NS
from os.path import join
from typing import Callable, Optional

import numpy as np
import torch
import yaml
from torch import FloatTensor as FT

import syngan.models as models
import syngan.utils as utils


def load_network_from_checkpoint(
    path_to_fits: str, task_name: str, **kwargs
) -> Callable:
    """
    Load trained network from checkpoint

    Args:
        path_to_fits: path to trained network.
        task_name: task name
        kwargs: keyword arguments for task.Factory
    """
    with open(join(path_to_fits, "config.yaml"), "r") as f:
        dic = {
            k: it["value"]
            for k, it in yaml.load(f, Loader=yaml.Loader).items()
            if "wandb" not in k
        }
        dic["init_gen_from_previous"] = False
        task = importlib.import_module("tasks.%s" % task_name)
        fact = task.Factory(**kwargs)
        fact.fconf = NS(**dic)

    chpt = torch.load(
        join(path_to_fits, "chpt_models.pt"), map_location=torch.device("cpu")
    )

    gen = fact.make_neural_network()
    if "dis" in dic["dis_type"]:
        netval = "gen"
    else:
        netval = "network"
    gen.load_state_dict(chpt["%s_state_dict" % netval])

    return gen


def make_new_data(
    net: str,
    rule: Callable,
    presyn_act: np.ndarray,
    wt_init_seed: Optional[int] = 42,
    init_weights: Optional[np.ndarray] = None,
    return_score: Optional[bool] = True,
    score_funcn: Optional[str] = None,
    PCs: Optional[np.ndarray] = None,
    n_presyn_neurs: Optional[int] = 3,
    n_postsyn_neurs: Optional[int] = 1,
    timesteps: Optional[int] = 200,
    update_rate: Optional[int] = 0.1,
    noise_amplitude: Optional[float] = 0.
):
    """
    Generate traces from learned rule.

    Args:
        net: rate network for simulation.
        rule: update rule.
        presyn_act: presynaptic activity with
                shape datasets x samples x n_presyn_neurons.
        wt_init_seed: seed for generating initial seed.
        init_weights: array of inital weights.
        return_score: if true, return score on weight.
        score_funcn: score function from utils.
        PCs: principal components.
        n_presyn_neurs: number of presynaptic neurons.
        n_postsyn_neurs: number of postsynaptic neurons.
        timesteps: number of timesteps to simulate.
        update_rate: update rate for rate network.
        noise_amplitude: noise amplitude for postsynaptic activity.
    """
    assert hasattr(models, net)
    if return_score:
        assert (
            (PCs is not None)
            and (score_funcn is not None)
            and (hasattr(utils, score_funcn))
        )
        score_funcn = getattr(utils, score_funcn)

    if init_weights is None:
        torch.manual_seed(wt_init_seed)
        init_weights = 0.1 * torch.abs(
            torch.randn(len(presyn_act), n_presyn_neurs, n_postsyn_neurs)
        )
    else:
        assert len(init_weights) == len(PCs)

    rate_net = getattr(models, net)(
        rule,
        update_rate=update_rate,
        n_presyn_neur=n_presyn_neurs,
        n_postsyn_neur=n_postsyn_neurs,
        timesteps=timesteps,
        noise_amplitude=noise_amplitude
    )

    weights = np.zeros((len(presyn_act), timesteps, n_presyn_neurs, n_postsyn_neurs))
    postsyn_act = np.zeros(
        (len(presyn_act), presyn_act.shape[1], timesteps, n_postsyn_neurs)
    )
    scores = np.zeros((len(presyn_act), timesteps))

    for i, (X, init_wt) in enumerate(zip(presyn_act, init_weights)):
        rate_net.wt_init.data = FT(init_wt)
        wt, y = rate_net(FT(X), return_weight=True)
        weights[i] = wt.data.numpy()[:, 1:]
        postsyn_act[i] = y.data.numpy()
        scores[i] = [
            score_funcn(FT(PCs[i: i + 1]), FT(w).unsqueeze(0)) for w in weights[i]
        ]

    if return_score:
        return weights, postsyn_act, scores

    return weights, postsyn_act
