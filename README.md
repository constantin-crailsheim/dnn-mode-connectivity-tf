# Mode Connectivity in TensorFlow

The content of this repository is based on the PyTorch implementation [dnn-mode-connectivity](https://github.com/timgaripov/dnn-mode-connectivity) by Timur Garipov, Pavel Izmailov, which is based on their paper [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026).

This project was part of the Summer 2022 course *Applied Deep Learning with TensorFlow and Pytorch* at the LMU Munich, supervised by [Dr. David Rügamer](https://www.slds.stat.uni-muenchen.de/people/ruegamer/).

# Setup

Setup a virtual environment

```shell
$ python -m venv .venv
$ source .venv/bin/activate
```

To install in development, if you want to make changes to the existing code

```shell
$ pip install -e .
```

or, if you want to execute the custom scripts and notebooks in 'showcase', install with additional dependencies

```shell
$ pip install -e .[showcase]
or 
$ pip install -e .[all]
```

# Structure

In `showcase/scripts`, you will find three different example scripts, for training and evaluation respectively. 

The `*_tensorflow.py` files display how you can use the architecture with the built in tensorflow functions `.fit()`and `.evaluate()`.

The `*_classification.py` files use a rather lower-level approach, and generate as well as save additional metrics which are used to display model progress in notebooks found in the `notebooks` folder.

The `*_regression.py` files train and evaluate the models on a simple regression dataset, which was originally used [here](https://github.com/wjmaddox/drbayes/blob/master/experiments/synthetic_regression/ckpts/data.npy).


# Training

## Base Models

```shell
$ python showcase/scripts/train_tensorflow.py --config base-model-1
or
$ python showcase/scripts/train_classification.py --config base-model-1
```
with a configuration like this in the `config.toml` file
```toml
[base-model-1]
dir = "results/MNIST_BasicCNN/checkpoints_base_model_1"
dataset = "mnist"
data-path = "datasets/"
model = "CNN"
epochs = 5
lr = 0.05
wd = 0.0005
seed = 1
```

You are also able to pass in arguments via the command line without specifying a config file.

## Curve
### With pretrained base models

You need to train the base models beforehand and match the epoch in init-start/end below.

```shell
$ python showcase/scripts/train_tensorflow.py --config curve-pretrained
or
$ python showcase/scripts/train_classification.py --config curve-pretrained
```
with 
```toml
[curve-pretrained]
dir = "results/MNIST_BasicCNN/checkpoints_curve"
dataset = "mnist"
data-path = "datasets/"
model = "CNN"
curve = "Bezier"
num-bends = 1
epochs = 10
lr = 0.05
wd = 0.0005
init-start = "results/MNIST_BasicCNN/checkpoints_base_model_1/model-weights-epoch5"
fix-start = true
init-end = "results/MNIST_BasicCNN/checkpoints_base_model_2/model-weights-epoch5"
fix-end = true
```

# Evaluate

```shell
$ python scripts/evaluate_tensorflow.py --config curve-evaluate
or
$ python scripts/evaluate_classification.py --config curve-evaluate
```
with
```toml
[curve-evaluate]
dir = "results/MNIST_BasicCNN/evaluation_curve"
dataset = "mnist"
data-path = "datasets/"
model = "CNN"
curve = "Bezier"
num-bends = 1
wd = 0.0005
ckpt = "results/MNIST_BasicCNN/checkpoints_curve/model-weights-epoch10"
init-linear = false
fix-start = true
fix-end = true
num-points = 11
```


# Tests

Run the test suite.

```shell
pip install -r requirements-dev.txt
pytest tests
```
