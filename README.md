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
```

# Structure

In `showcase/scripts`, you will find three different example scripts, for training and evaluation respectively. 

The `*_tensorflow.py` files display how you can use the architecture with the built in tensorflow functions `.fit()`and `.evaluate()`.

The `*_classification.py` files use a rather lower-level approach, and generate as well as save additional metrics which are used to display model progress in notebooks found in the `notebooks` folder.

The `*_regression.py` files train and evaluate the models on a simple regression dataset, which was originally used [here](https://github.com/wjmaddox/drbayes/blob/master/experiments/synthetic_regression/ckpts/data.npy).


# Training

## Base Models

There are three different models defined:
- CNN: CNN model for classification of MNIST dataset.
- CNNBN: CNN model with batch normalization for classification of MNIST dataset.
- MLP: MLP with one hidden layer for regression mainly for visualisation.

The base models can be trained by (here for CNN):
```shell
$ python showcase/scripts/train_tensorflow.py --config cnn-base-model-1
or
$ python showcase/scripts/train_classification.py --config cnn-base-model-1
```
with a configuration like this in the `config.toml` file
```toml
[cnn-base-model-1]
dir = "results/MNIST_CNN/checkpoints_base_model_1"
dataset = "mnist"
data-path = "datasets/"
model = "CNN"
epochs = 10
lr = 0.05
wd = 0.0005
seed = 1
```

You are also able to pass in arguments via the command line without specifying a config file.

## Curve
### With pretrained base models

You need to train the two base models beforehand and match the epoch in init-start/end below.

```shell
$ python showcase/scripts/train_tensorflow.py --config cnn-curve-pretrained
or
$ python showcase/scripts/train_classification.py --config cnn-curve-pretrained
```
with 
```toml
[cnn-curve-pretrained]
dir = "results/MNIST_CNN/checkpoints_curve"
dataset = "mnist"
data-path = "datasets/"
model = "CNN"
curve = "Bezier"
num-bends = 1
epochs = 10
lr = 0.05
wd = 0.0005
init-start = "results/MNIST_CNN/checkpoints_base_model_1/model-weights-epoch10"
fix-start = true
init-end = "results/MNIST_CNN/checkpoints_base_model_2/model-weights-epoch10"
fix-end = true
```

### Resume training

Training can be resumed from a given checkpoint.

```shell
$ python showcase/scripts/train_tensorflow.py --config cnn-curve-resume
or
$ python showcase/scripts/train_classification.py --config cnn-curve-resume
```

with 
```toml
dir = "results/MNIST_BasicCNN/checkpoints_curve"
dataset=mnist
data-path=datasets,
ckpt=results/MNIST_BasicCNN/checkpoints_curve/model-weights-epoch10
resume-epoch=11
model=CNN
curve=Bezier
num-bends=1
epochs=15
lr=0.05
wd=0.0005
fix-start = True
fix-end = True
```

# Evaluate

The curve can be evaluted for a certain number of equidistant points on curve.

```shell
$ python scripts/evaluate_tensorflow.py --config cnn-curve-evaluate
or
$ python scripts/evaluate_classification.py --config cnn-curve-evaluate
```
with
```toml
[cnn-curve-evaluate]
dir = "results/MNIST_CNN/evaluation_curve"
dataset = "mnist"
data-path = "datasets/"
model = "CNN"
curve = "Bezier"
num-bends = 1
wd = 0.0005
ckpt = "results/MNIST_CNN/checkpoints_curve/model-weights-epoch10"
init-linear = false
fix-start = true
fix-end = true
num-points = 11
```

Alternatively a specific point on the curve can be evaluated.

```shell
$ python scripts/evaluate_tensorflow.py --config cnn-curve-evaluate-point
or
$ python scripts/evaluate_classification.py --config cnn-curve-evaluate-point
```
with
```toml
[cnn-curve-evaluate-point]
dir = "results/MNIST_CNN/evaluation_curve"
dataset = "mnist"
data-path = "datasets/"
model = "CNN"
curve = "Bezier"
num-bends = 1
wd = 0.0005
ckpt = "results/MNIST_CNN/checkpoints_curve/model-weights-epoch10"
init-linear = false
fix-start = true
fix-end = true
point-on-curve = 0.5
```

# Tests

Run the test suite.

```shell
pip install -r requirements-dev.txt
pytest tests
```

# Comments

There are some issues which may cause confusion:
- The train loss might be larger than the test loss. This can happen, since the train loss is computed as average over all minibatches over the whole epoch, while the model is still improving. The test loss, on the other hand, is computed for the best model at the end of the epoch.
- The test loss of the fixed corner points might not correspond to the test loss of the last epoch of the base models. This can happen, since for evaluation the moving mean/variance have to be recomputed for every point on curve for the final model and thus might not exactly correspond to the moving mean/variance computed during training.