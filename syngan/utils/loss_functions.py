from typing import Optional

import torch


def pc_norm(
    input_pc: torch.tensor,
    weight: Optional[torch.tensor] = None,
) -> torch.tensor:
    """
    Compute loss based on first PC of presynaptic activity.

    Args:
        input_pc: First principal component of the presynaptic activity of
                    shape batch_size x n_presyn_neur
        weight: synaptic weight to calculate loss.
    """
    # TODO ONLY works when there is one postsynaptic neuron
    return torch.norm(
        torch.abs(input_pc[..., 0, :].squeeze()) - torch.abs(weight[..., 0].squeeze())
    )


def pc_norm_all(
    input_pc: torch.tensor,
    weight: Optional[torch.tensor] = None,
) -> torch.tensor:
    """
    Compute loss based on all PCs of presynaptic activity.

    Args:
        input_pc: principal components of the presynaptic activity of
                    shape batch_size x n_presyn_neur
        weight: synaptic weight to calculate loss.
    """
    # TODO ONLY works when there is one postsynaptic neuron
    return torch.norm(
        torch.abs(input_pc.squeeze(0)) - torch.abs(weight.squeeze(0)), -1
    ).mean()


def mse(gt_activity: torch.tensor, gen_activity: torch.tensor) -> None:
    """
    Compute mean-squared error between groundtruth and generated
    postsynaptic activity.

    Args:
        gt_activity: groundtruth post-synaptic activity.
        gen_activity: generated post-synaptic activity.
    """
    return ((gt_activity - gen_activity) ** 2).mean()


def wasserstein(d_fake: torch.tensor,
                d_real: Optional[torch.tensor] = None) -> torch.tensor:
    """
    Calculate Wasserstein distance.

    Args:
        d_fake: discriminator output for fake data.
        d_real: discriminator output for real data.
    """
    if d_real is None:
        return - d_fake
    else:
        return - (d_real - d_fake)


def cross_entropy(d_fake: torch.tensor,
                  d_real: Optional[torch.tensor] = None) -> torch.tensor:
    """
    Calculate binary cross entropy(BCE) loss.

    Args:
        d_fake: discriminator output for fake data.
        d_real: discriminator output for real data.
    """
    if d_real is None:
        return torch.log(1 - d_fake + 1e-8)
    else:
        return - torch.log(d_real + 1e-8) -\
                 torch.log(1 - d_fake + 1e-8)


def flip_cross_entropy(d_fake: torch.tensor,
                       d_real: Optional[torch.tensor] = None) -> torch.tensor:
    """
    Calculate BCE loss with flipped targets i.e d_real -> 0, d_fake -> 1.

    Args:
        d_fake: discriminator output for fake data.
        d_real: discriminator output for real data.
    """
    if d_real is None:
        return torch.log(d_fake)
    else:
        return - torch.log(d_fake) - torch.log(1 - d_real)


def nonsat_cross_entropy(d_fake: torch.tensor,
                         d_real: Optional[torch.tensor] = None
                         ) -> torch.tensor:
    """
    Calculate non-saturating BCE loss.

    Args:
        d_fake: discriminator output for fake data.
        d_real: discriminator output for real data.
    """
    if d_real is None:
        return - flip_cross_entropy(d_fake)
    else:
        return cross_entropy(d_fake, d_real)


loss_dict = {"wasserstein": wasserstein,
             "cross_entropy": cross_entropy,
             "flip_cross_entropy": flip_cross_entropy,
             "nonsat_cross_entropy": nonsat_cross_entropy
             }
