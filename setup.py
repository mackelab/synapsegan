#!/usr/bin/env python

from setuptools import find_packages, setup

package_name = "syngan"
version = '1.0'
exclusions = ["notebooks"]

_packages = find_packages(exclude=exclusions)

_base = ["numpy", "matplotlib", "scipy", "seaborn", "sklearn", "torch", "pyyaml", "spikegan@git+https://github.com/mackelab/spikegan#egg=spikegan"]

setup(name=package_name,
      version=version,
      description="Generative adversarial networks for meta-learning plasticity rules",
      author="Poornima Ramesh",
      author_email="poornima.ramesh@uni-tuebingen.de",
      url="https://github.com/mackelab/synapsegan",
      packages=['syngan'],
      _install_requires=(_base)
)
