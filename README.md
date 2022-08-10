# Subspace inference in TensorFlow
Implementation of subspace inference in tensorflow based on PyTorch code.

Link to orignal Repo:

https://github.com/timgaripov/dnn-mode-connectivity

# Setup

Install in development mode

```shell
python -m venv .venv
source .venv/bin/activate

pip install -e .
```

# Run

## Base

```shell
$ train-from-config base-model-1
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
transform = "CNN"
seed = 1
```
or
```shell
$ train --dir=results/MNIST_BasicCNN/checkpoints_base_model_1 \
    --dataset=mnist \
    --data-path=datasets/ \
    --model=CNN \
    --epochs=5 \
    --lr=0.01 \
    --wd=1e-4 \
    --transform=CNN 
```

## Curve
### With pretrained base models

You need to train the base models beforehand and match the epoch in init-start/end below.

```shell
$ train-from-config curve-pretrained
```
with 
```toml
[curve-pretrained]
dir = "results/MNIST_BasicCNN/checkpoints_curve"
dataset = "mnist"
data-path = "datasets/"
model = "CNN"
curve = "Bezier"
transform = "CNN"
num-bends = 1
epochs = 10
lr = 0.05
wd = 0.0005
init-start = "results/MNIST_BasicCNN/checkpoints_base_model_1/model-weights-epoch5"
fix-start = true
init-end = "results/MNIST_BasicCNN/checkpoints_base_model_2/model-weights-epoch5"
fix-end = true
```
or 
```shell
$ train --dir=results/MNIST_BasicCNN/checkpoints_curve \
 --dataset=mnist \
 --data-path=datasets/ \
 --model=CNN \
 --curve=Bezier \
 --transform=CNN \
 --num-bends=1 \
 --epochs=3 \
 --fix-start \
 --init-start=results/MNIST_BasicCNN/checkpoints_base_model_1/model-weights-epoch5 \
 --fix-end \
 --init-end=results/MNIST_BasicCNN/checkpoints_base_model_2/model-weights-epoch5
```

# Evaluate

```shell
$ evaluate-from-config curve-evaluate
```
with
```toml
[curve-evaluate]
dir = "results/MNIST_BasicCNN/evaluation_curve"
dataset = "mnist"
data-path = "datasets/"
model = "CNN"
curve = "Bezier"
transform = "CNN"
num-bends = 1
wd = 0.0005
ckpt = "results/MNIST_BasicCNN/checkpoints_curve/model-weights-epoch10"
init-linear = false
fix-start = true
fix-end = true
num-points = 11
```
or 
```shell
$ evaluate --dir=results/MNIST_BasicCNN/evaluation_curve \
  --dataset=mnist \
  --data-path=datasets/ \
  --model=CNN \
  --curve=Bezier \
  --transform=CNN \
  --num-bends=1 \
  --wd=5e-4 \
  --ckpt=results/MNIST_BasicCNN/checkpoints_curve/model-epoch10 \
  --init-linear-off \
  --fix-start \
  --fix-end \
  --num-points=11
```

# Tests

```shell
pip install -r requirements-dev.txt
pytest pytest tests --forked
```
