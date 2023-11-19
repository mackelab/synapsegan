"""Pipeline for experiments with Oja's Rule with added noise."""
from tasks.oja_net_small import Factory as AF


class Factory(AF):
    def __init__(
            self,
            path_to_fit_conf="./tasks/oja_net_noise_small/",
            path_to_sim_conf="../data/oja_net_noise_small/",
            resume=False,
            resume_dir=None) -> None:
        """Set up factory for Oja's rule and small net with noise pipeline."""
        super(Factory, self).__init__(
            path_to_fit_conf, path_to_sim_conf, resume, resume_dir)
