# Mode Connectivity in TensorFlow

The content of this repository is based on the PyTorch implementation [dnn-mode-connectivity](https://github.com/timgaripov/dnn-mode-connectivity) by Timur Garipov, Pavel Izmailov, which is based on their paper [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026).

This project was part of the Summer 2022 course *Applied Deep Learning with TensorFlow and Pytorch* at the LMU Munich, supervised by [Dr. David Rügamer](https://www.slds.stat.uni-muenchen.de/people/ruegamer/).

# Setup

Setup a virtual environment

```shell
$ python -m venv .venv
$ source .venv/bin/activate
```

To install in development mode (if you want to make changes to the existing code)

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
[cnn-curve-resume]
dir = "results/MNIST_BasicCNN/checkpoints_curve"
dataset="mnist"
data-path="datasets/"
ckpt="results/MNIST_BasicCNN/checkpoints_curve/model-weights-epoch10"
resume-epoch=11
model="CNN"
curve="Bezier"
num-bends=1
epochs=15
lr=0.05
wd=0.0005
fix-start = true
fix-end = true
```

# Evaluate

The curve can be evaluted for a certain number of equidistant points on curve.

```shell
$ python showcase/scripts/evaluate_tensorflow.py --config cnn-curve-evaluate
or
$ python showcase/scripts/evaluate_classification.py --config cnn-curve-evaluate
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
file-name-appendix = "_epoch10"
```

Alternatively a specific point on the curve can be evaluated.

```shell
$ python showcase/scripts/evaluate_tensorflow.py --config cnn-curve-evaluate-point
or
$ python showcase/scripts/evaluate_classification.py --config cnn-curve-evaluate-point
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
save-evaluation = false
```

# Tests

Run the test suite.

```shell
pip install -r requirements-dev.txt
pytest tests
```

# Execute config script

Execute all scripts for the CNNs on built in tensorflow functions.

```shell
bash execute_config_tf.sh
```

However, to visualise the results and get results for the regression problem, the custom scripts need to be executed.

Execute all scripts on built in custom training/evaluation routines:

```shell
bash execute_config_custom.sh
```

# Comments

There are some issues which may cause confusion:
- The train loss might be larger than the test loss. This can happen, since the train loss is computed as average over all minibatches over the whole epoch while the model is still improving. The test loss, on the other hand, is computed for the best model at the end of the epoch.
- The test loss for the fixed corner points in the curve model and the base models is different, since the curve model has more weights and therefore a different penalty term. Also the loss differs between the initialized and trained curve model, since the inner curve weights are different and thus the regularized loss evaluates differently. For the classification task we can see that the corner points of the curve model correspond to the base models, since the accuracies are the same. For the regression task, we used the unregularized loss for evaluation and testing to show that the corner points of the curve model and base models are in fact the same.
- For the CurveNet with BatchNorm, no testing is possible during the training mode, since the moving mean/variance was computed based on the randomly sampled points on the curve during training and thus would not fit to the randomly sampled points on the curve during testing.
- For the CurveNet with BatchNorm, the test loss of the fixed corner points might not correspond to the test loss of the last epoch of the base models. This can happen, since for evaluation the moving mean/variance have to be recomputed for every point on curve for the final model and thus might not exactly correspond to the moving mean/variance computed during training.
- Although this repository is layed out to work with the Apple M1 architecture, it won't work properly with the current tensorflow-metal version 0.5. This is because random number generation is broken in this version, thus training with different random generated points on curve is impossible. Apple seems to be aware of this issue and working on it though. See this [thread](https://developer.apple.com/forums/thread/697057).
