import os

from unittest import mock

import keras
import numpy as np
import pytest
import tensorflow as tf

from mode_connectivity.models.cnn import CNN, CNNBase
from mode_connectivity.models.mlp import MLP

from mode_connectivity.argparser import Arguments, parse_train_arguments

from mode_connectivity.utils import (
    learning_rate_schedule,
    adjust_learning_rate,
    get_architecture,
    save_weights,
    get_model,
    get_epoch
)

@pytest.fixture
def checkpoints_dir(tmpdir):
    return tmpdir.mkdir("checkpoints")


@pytest.fixture
def basic_model() -> keras.Model:
    class ModelFactory:
        def get(self, fitted: bool = False):
            inputs = tf.keras.layers.Input(shape=(10,), name="my_input")
            outputs = tf.keras.layers.Dense(10)(inputs)
            model = tf.keras.Model(inputs, outputs)
            model.compile(tf.keras.optimizers.SGD(), loss="mse")
            if not fitted:
                return model
            tensors = tf.random.uniform((10, 10)), tf.random.uniform((10,))
            dataset = tf.data.Dataset.from_tensor_slices(tensors).repeat().batch(1)
            model.fit(dataset, epochs=1, steps_per_epoch=10)
            return model

    return ModelFactory()


@pytest.fixture
def basic_optimizer() -> keras.optimizers.Optimizer:
    return tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.1)


testdata_lr_schedule = [
    (1, 1, 20, 1), 
    (1, 10, 20, 1),
    (1, 14, 20, 0.505),
    (1, 18, 20, 0.01),
    (1, 20, 20, 0.01)
]

@pytest.mark.parametrize("base_lr, epoch, total_epochs, expected_lr", testdata_lr_schedule)
def test_learning_rate_schedule(base_lr, epoch, total_epochs, expected_lr):
    lr = learning_rate_schedule(base_lr, epoch, total_epochs)
    assert np.allclose(lr, expected_lr)

@pytest.mark.parametrize("lr", [0.01, 0.05, 0.2, 0.5, 1.0])
def test_adjust_learning_rate(lr, basic_optimizer):
    adjust_learning_rate(basic_optimizer, lr)
    assert basic_optimizer.lr == lr

def test_save_weights(checkpoints_dir, basic_model, basic_optimizer):
    assert not os.path.isfile(
        os.path.join(checkpoints_dir, "model-weights-epoch1.index")
    )
    model = basic_model.get()
    save_weights(
        directory=checkpoints_dir,
        epoch=1,
        model=model
    )
    assert os.path.isfile(
        os.path.join(checkpoints_dir, "model-weights-epoch1.index")
    )

# TODO Change to CNN and CNNBase
@pytest.mark.parametrize("model_name", ["CNN", "MLP"])
def test_get_architecture(model_name):
    architecture = get_architecture(model_name)
    if model_name == "CNN":
        assert issubclass(architecture, CNN)
    elif model_name == "MLP":
        assert issubclass(architecture, MLP)

# TODO Change to CNN and CNNBase
def test_get_regular_model_CNN():
    arguments = [
        "python",
        "--model",
        "CNN"
        ]
    with mock.patch("sys.argv", arguments):
        args = parse_train_arguments()

    architecture = get_architecture(model_name=args.model)
    model = get_model(
        architecture=architecture,
        args=args,
        num_classes=10,
        input_shape=(None, 28, 28, 1)
    )

    assert len(model.layers) == 2
    assert len(model.layers[0].layers) == 9
    assert len(model.layers[1].layers) == 5

def test_get_regular_model_MLP():
    arguments = [
        "python",
        "--model",
        "MLP"
        ]
    with mock.patch("sys.argv", arguments):
        args = parse_train_arguments()

    architecture = get_architecture(model_name=args.model)
    model = get_model(
        architecture=architecture,
        args=args,
        num_classes=None,
        input_shape=(None,2)
    )

    assert len(model.layers) == 1
    assert len(model.layers[0].layers) == 2

def test_get_epoch_not_resume():
    arguments = [
        "python",
        "--model",
        "CNN",
        ]
    with mock.patch("sys.argv", arguments):
        args = parse_train_arguments()
    start_epoch = get_epoch(args)
    assert start_epoch == 1

def test_get_epoch_resume():
    arguments = [
        "python",
        "--model",
        "CNN",
        "--ckpt",
        "SomeCheckpoint",
        "--resume-epoch",
        "21"
        ]
    with mock.patch("sys.argv", arguments):
        args = parse_train_arguments()
    start_epoch = get_epoch(args)
    assert start_epoch == 21
