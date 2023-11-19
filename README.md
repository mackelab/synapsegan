# Indistinguishable network dynamics can emerge from unalike plasticity rules
---
Code package `synapsegan` implementing method and experiments described in the associated manuscript  ["Indistinguishable network dynamics can emerge from unalike plasticity rules"](https://www.biorxiv.org/content/10.1101/2023.11.01.565168v1)

___
### Installation

With a working Python environment, install `synapsegan` using `pip`:
```
pip install "git+https://github.com/mackelab/synapsegan"
```
___
### Experiments

The paper describes results for experiments with Oja's Rule and a rate network, and fitting GANs to data generated from the former, with different parametrizations of the plasticity rule.

Code for setting up GAN networks, any other pre-/post-processing code, and the hyperparameter settings for the experiments is available inside `tasks/`.

To reproduce the experiments, change the configuration settings for the appropriate experiment in `tasks/`, and use the `run.py` script from the repository's root directory (note that this relies on `wandb` to log experiments):
```
python run.py --task_name TASK_NAME
```
where `TASK_NAME` can be `oja_net_small`, `oja_net_noise_small` OR `oja_net_big`

Note that we **do not** provide training data for these experiments. However, users can generate training data using `syngan.utils.make_training_data.make_training_test_datasets`.
___
### Figures

Code to reproduce the figures in the paper is available in `plotting_code`. However this is only code: the data required to run it will be provided upon request.
