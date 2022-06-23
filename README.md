# Subspace inference in TensorFlow
Implementation of subspace inference in tensorflow based on PyTorch code..

# Setup

Install in development mode

```console
pip install -e .
```

# Run

## Base

```console
train --dir=results/MNIST_BasicCNN/checkpoints_model_1 \
    --dataset=mnist \
    --data-path=datasets/ \
    --model=CNN \
    --epochs=3 \
    --lr=0.01 \
    --wd=1e-4 \
    --transform=CNN 
```

## Curve
### With pretrained base models

You need to train the base models beforehand and match the epoch in init-start/end below.

```console
train --dir=results/MNIST_BasicCNN/checkpoints_curve \
 --dataset=mnist \
 --data-path=datasets/ \
 --model=CNN \
 --curve=Bezier \
 --transform=CNN \
 --num-bends=3 \
 --epochs=3 \
 --fix-start \
 --init-start=results/MNIST_BasicCNN/checkpoints_model_1/model-weights-epoch3 \
 --fix-end \
 --init-end=results/MNIST_BasicCNN/checkpoints_model_2/model-weights-epoch3
```

# Tests

```console
pip install -r requirements-dev.txt
pytest tests
```