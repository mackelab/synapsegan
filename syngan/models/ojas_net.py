"""Rate network with linear updates."""

import copy
from typing import Optional

import torch

from . import AbstractNeuralNet, AbstractRule


class OjaNet(AbstractNeuralNet):
    """Rate net with Oja's Rule."""

    def __init__(
        self,
        update_rule: AbstractRule,
        update_rate: Optional[float] = 0.1,
        n_presyn_neur: Optional[int] = 3,
        n_postsyn_neur: Optional[int] = 1,
        timesteps: Optional[int] = 200,
        noise_amplitude: Optional[float] = 0.0,
    ):
        r"""
        Set up neural net to work with Oja's Rule.
        $ y_i = \sum_{j}\omega_{ij}x_j$

        Args:
            update_rule: class containing parameterised update rule.
            update_rate: learning rate $\eta$ for update rule.
            n_presyn_neur: number of presynaptic neurons.
            n_postsyn_neur: number of postsynaptic neurons.
            timesteps: number of timesteps for which to simulate postsynaptic
                       activity.
            noise_amplitude: multiplicative factor for added Gaussian noise to
                            post-synaptic activity
        """
        super(OjaNet, self).__init__(update_rule=update_rule, update_rate=update_rate)
        self.n_presyn_neur = n_presyn_neur
        self.n_postsyn_neur = n_postsyn_neur
        self.timesteps = timesteps
        self.noise_amplitude = noise_amplitude

        # Random init for synaptic weight matrix
        self.wt_init = 1e-1 * torch.randn(1, self.n_presyn_neur, self.n_postsyn_neur)

    def update_synaptic_weight(
        self,
        weight: torch.tensor,
        presyn_activity: torch.tensor,
        postsyn_activity: torch.tensor
    ) -> None:
        """
        Update synaptic weights.

        Args:
            presyn_activity: activity of presynaptic neurons.
            postsyn_activity: activity of postsynaptic neurons.
        """
        # Need to detach previous weight from computational graph while
        # adding new weights to computational graph
        update = self.update_rule(
            weight, presyn_activity, postsyn_activity
        ).reshape(-1, self.n_presyn_neur, self.n_postsyn_neur)
        # Update weights
        return weight + self.update_rate * update
        # norm = torch.max(self.current_weight).detach()
        # self.current_weight /= norm

    def _fwd_pass(self, weight, presyn_activity):
        presyn_activity = presyn_activity.reshape(-1, 1, self.n_presyn_neur)
        noise = self.noise_amplitude * torch.randn(
            len(presyn_activity), 1, self.n_postsyn_neur
        ).to(presyn_activity.device)
        return (
            torch.matmul(
                presyn_activity, weight
            )
            + noise
        )

    def forward(
        self,
        presyn_act: torch.tensor,
        timesteps: Optional[int] = None,
        return_weight: Optional[bool] = False,
    ) -> torch.tensor:
        """
        Forward pass to get post-synaptic activity.

        Args:
            presyn_act: activity of presynaptic neurons.
            timesteps: number of timesteps to simulate postsynaptic activity.
            return_weight: if True, track and return synaptic weights and
                           outputs.
        """
        # Last dimension of presyn activity should be number
        # of presyn neurons
        assert presyn_act.shape[-1] == self.n_presyn_neur

        # Initialise weights and activity
        weight = copy.deepcopy(self.wt_init).to(presyn_act.device)
        if timesteps is None:
            timesteps = self.timesteps

        postsyn_act = self._fwd_pass(weight, presyn_act)

        postsyn_act_time = []

        if return_weight:
            weights = [weight.reshape(-1, 1,
                                      self.n_presyn_neur,
                                      self.n_postsyn_neur)
                       ]

        # Run model forward
        t = 0
        while t < timesteps:
            weight = self.update_synaptic_weight(weight,
                                                 presyn_act,
                                                 postsyn_act
                                                 ).reshape(-1, 1,
                                      self.n_presyn_neur,
                                      self.n_postsyn_neur)
            postsyn_act = self._fwd_pass(weight, presyn_act
                                         ).reshape(-1, 1,
                                                   self.n_postsyn_neur)

            # norm = torch.norm(act_init).detach()
            postsyn_act_time.append(postsyn_act)  # / (norm + 1e-3)

            if return_weight:
                weights.append(weight)

            # Check if activity is finite, else stop fwd pass
            if (
                not torch.isfinite(postsyn_act).all()
                and not torch.isfinite(weight).all()
                and (torch.max(weight.data) < 1e5 / len(presyn_act))
            ):
                print("Activity not finite. Stopped at timestep %d" % t)
                return torch.cat(postsyn_act_time, 1)
            t += 1

        if return_weight:
            return torch.cat(weights, 1), torch.cat(postsyn_act_time, 1)

        return torch.cat(postsyn_act_time, 1)
