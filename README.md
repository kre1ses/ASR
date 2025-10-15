# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a template for solving ASR task with PyTorch. This template branch is a part of the [HSE DLA course](https://github.com/markovka17/dla) ASR homework. Some parts of the code are missing (or do not follow the most optimal design choices...) and students are required to fill these parts themselves (as well as writing their own models, etc.).

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw1_asr).

## Installation

Follow these steps to install the project:

0. (Highly recomended) Create and activate new environment using `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

   If you are using Colab or Kaggle make sure that you have correct path to requirements.txt
   Also, using correct directory will skip BPE training

2. Install `pre-commit`:

   ```bash
   pre-commit install
   ```

3. If you want to log model performance using Cometml:

   ```bash
   export COMET_API_KEY=YOUR_API_KEY
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

If you want to use distributed training, run the following command:

```bash
accelerate launch train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
