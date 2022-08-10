import os

import keras
import numpy as np
import pytest
import tensorflow as tf

from showcase.models.cnn import CNN
from showcase.models.mlp import MLP

from showcase.utils import (
    learning_rate_schedule,
    adjust_learning_rate,
    get_architecture,
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
    (1, 20, 20, 0.01),
]


@pytest.mark.parametrize(
    "base_lr, epoch, total_epochs, expected_lr", testdata_lr_schedule
)
def test_learning_rate_schedule(base_lr, epoch, total_epochs, expected_lr):
    lr = learning_rate_schedule(base_lr, epoch, total_epochs)
    assert np.allclose(lr, expected_lr)


@pytest.mark.parametrize("lr", [0.01, 0.05, 0.2, 0.5, 1.0])
def test_adjust_learning_rate(lr, basic_optimizer):
    adjust_learning_rate(basic_optimizer, lr)
    assert basic_optimizer.lr == lr


# TODO test save weights

# TODO check on how to properly test it
# @pytest.mark.parametrize("model_name", ["CNN", "MLP"])
# def test_get_architecture(model_name):
#     architecture = get_architecture(model_name)
#     if model_name == "CNN":
#         assert isinstance(architecture, CNN)
#     elif model_name == "MLP":
#         assert isinstance(architecture, MLP)


# TODO Write tests for util functions.

#
#     def test_save_checkpoint(self, checkpoints_dir, basic_model, basic_optimizer):
#         assert not os.path.isfile(
#             os.path.join(checkpoints_dir, "checkpoint-epoch1-1.index")
#         )
#         model = basic_model.get()
#         save_checkpoint(
#             directory=checkpoints_dir,
#             epoch=1,
#             model=model,
#             optimizer=basic_optimizer,
#             name="checkpoint",
#         )
#         assert os.path.isfile(
#             os.path.join(checkpoints_dir, "checkpoint-epoch1-1.index")
#         )

#     def test_load_checkpoint(self, checkpoints_dir, basic_model, basic_optimizer):
#         model = basic_model.get(fitted=True)
#         new_model = basic_model.get()

#         weights_model = model.get_weights()
#         weights_new_model = new_model.get_weights()
#         for i in range(len(weights_model)):
#             assert not np.allclose(weights_model[i], weights_new_model[i])

#         save_checkpoint(
#             directory=checkpoints_dir,
#             epoch=1,
#             model=model,
#             optimizer=basic_optimizer,
#             name="checkpoint",
#         )

#         ckpt_path = os.path.join(checkpoints_dir, "checkpoint-epoch1-1")
#         epoch = load_checkpoint(ckpt_path, model=new_model, optimizer=basic_optimizer)
#         assert epoch == 2

#         weights_model = model.get_weights()
#         weights_new_model = new_model.get_weights()
#         for i in range(len(weights_model)):
#             assert np.allclose(weights_model[i], weights_new_model[i])
