from syngan.models import OjaNet, OjaRule
import torch
import numpy as np
from sklearn.decomposition import PCA
from torch.distributions import MultivariateNormal as MVN
import matplotlib.pyplot as plt
from .loss_functions import pc_norm

# Define covariance matrix
D3 = np.diag([1, 0.5, 1e-6])
D39 = np.diag([1., 0.9, 0.9, 0.9,
               0.8, 0.8, 0.8, 0.7, 0.7, 0.7,
               0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
               0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
               1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
D_dict = {"3": D3, "39": D39}


def make_presynaptic_activity(n_samples: int = 100, n_presyn_neur: int = 3) -> torch.Tensor:
    """
    Make presynaptic activity from defined covariance matrix.

    Args:
        n_samples: number of samples
        n_presyn_neur: number of presynaptic neurons
    """
    D = D_dict[str(n_presyn_neur)]
    A = np.random.rand(n_presyn_neur, n_presyn_neur)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q = -Q
    Cov = Q.T.dot(D.dot(Q))
    return MVN(torch.zeros(n_presyn_neur), covariance_matrix=torch.FloatTensor(Cov)
               ).sample(torch.Size([n_samples]))


def get_pcs(presyn_act: torch.Tensor) -> torch.Tensor:
    """
    Get principal components of presynaptic activity.

    Args:
        presyn_act: presynaptic activity
    """
    pca = PCA()
    pca.fit(presyn_act)
    return pca.components_


def make_training_test_datasets(n_presyn_neurs: int = 3,
                                n_postsyn_neurs: int = 1,
                                update_rate: float = 0.1,
                                n_datasets: int = 100,
                                n_samples: int = 100,
                                timesteps: int = 200,
                                noise_amplitude: int = 0.) -> tuple:
    """
    Make training and test datasets for Oja Net.

    Args:
        n_presyn_neurs: number of presynaptic neurons. Default is 3, alternative is 39.
        n_postsyn_neurs: number of postsynaptic neurons. Default is 1
        update_rate: learning rate for synaptic weight update. Default is 0.1
        n_datasets: number of datasets (with different presynaptic activity / principal components). Default is 100
        n_samples: number of pre-synaptic activity samples per dataset. Default is 100
        timesteps: number of timesteps for which to simulate postsynaptic activity assuming same presynaptic activity for every time step. Default is 200
        noise_amplitude: multiplicative factor for added Gaussian noise to post-synaptic activity
    """
    X, pcas, Y, W = [], [], [], []

    ojanet = OjaNet(OjaRule(n_presyn_neurs, n_postsyn_neurs),
                    update_rate,
                    n_presyn_neurs,
                    n_postsyn_neurs,
                    timesteps=timesteps,
                    noise_amplitude=noise_amplitude)

    for ndat in range(n_datasets):
        x = make_presynaptic_activity(n_samples=n_samples,
                                    n_presyn_neur=n_presyn_neurs)
        weights, y = ojanet.forward(x, return_weight=True)
        pc = get_pcs(x.reshape(-1, n_presyn_neurs))
        X.append(x.unsqueeze(0))
        pcas.append(torch.FloatTensor(pc).unsqueeze(0))
        Y.append(y.unsqueeze(0).data)
        W.append(weights.unsqueeze(0).data)

    scores = [[pc_norm(pca.reshape(1, n_presyn_neurs, n_presyn_neurs), ww.reshape(1, n_presyn_neurs, 1)).data.numpy().tolist()
           for ww in wt.squeeze(0)]
          for (wt, pca) in zip(W, pcas)
         ]

    X = torch.cat(X, 0)
    pcas = torch.cat(pcas, 0)
    Y = torch.cat(Y, 0)
    W = torch.cat(W, 0)
    scores = torch.FloatTensor(np.array(scores))

    return X, pcas, Y, W, scores
