# StochFlow: A Python Library for Stochastic Interpolant Models

> DISCLAIMER: Package is in continued development.

StochFlow is a Python library that implements the stochastic interpolant models and algorithms described in the paper *Stochastic Interpolants via Diffusive and Flow-based Processes* by Chen et al **[preprint]**. The library provides a set of tools for designing generative models using stochastic processes associated with time-evolving probability density functions as well as the forward and backward Fokker-Planck equations.

<p align="center">
  <img src="Assets/description_pic.png" />
</p>

## Features

- Implementation of the stochastic interpolant models and algorithms described in the paper.
- Numerical methods for solving the forward and backward Fokker-Planck equations, including the Crank-Nicolson method and the Strang splitting method.
- Tools for generating samples from a given probability distribution using the forward and backward generative models.
- Support for both flow-based and diffusion-based methods for generative modelling.
- Examples and tutorials demonstrating the use of the library.

## Installation

To install StochFlow, simply run the following command:

```bash
pip3 install stochflow
pip3 install $(grep -ivE "torch|torchvision|scikit-learn" requirements.txt)
```
It may be necessary to manually install PyTorch and scikit-learn.

**[PyTorch](https://pytorch.org/get-started/locally)** ➞ Select the stable build for your operating system, your preferred package manager, and compute platform depending on your GPU capabilities. Then, run the recommended commands to install PyTorch locally.

**[scikit-learn](https://scikit-learn.org/stable/install)** ➞ Select your operating system/packager, and run the provided command to install the latest official release.

## Usage

To use StochFlow, simply import the relevant modules and functions:

```python
import stochflow as sf

# Define the initial and final probability density functions
rho_0 = sf.GaussianMixtureModel(...)
rho_1 = sf.GaussianMixtureModel(...)

# Solve the forward Fokker-Planck equation to generate samples from rho_1
samples_forward = sf.forward_generative_model(rho_0, rho_1, ...)

# Solve the backward Fokker-Planck equation to generate samples from rho_0
samples_backward = sf.backward_generative_model(rho_0, rho_1, ...)
```

The library provides a range of options and parameters for customizing the generative models and numerical methods used to solve the Fokker-Planck equations. See the documentation and examples for more details.

## Examples

The library includes several examples and tutorials demonstrating the use of the stochastic interpolant models and algorithms. These examples cover a range of applications, including image generation, time series modeling, and data augmentation.

## Contributing

Contributions to StochFlow are welcome! If you find a bug or have a feature request, please open an issue on the GitHub repository. If you would like to contribute code, please fork the repository and submit a pull request.

## License

StochFlow is released under the MIT License. See the LICENSE file for more details.
