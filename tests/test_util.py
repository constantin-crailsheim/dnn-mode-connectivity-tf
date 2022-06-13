import os

import keras
import numpy as np
import pytest
import tensorflow as tf
from mode_connectivity.utils import load_checkpoint, save_checkpoint


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


class TestUtil:
    def test_save_checkpoint(self, checkpoints_dir, basic_model, basic_optimizer):
        assert not os.path.isfile(os.path.join(checkpoints_dir, "checkpoint-1.index"))
        model = basic_model.get()
        save_checkpoint(
            directory=checkpoints_dir,
            epoch=1,
            model=model,
            optimizer=basic_optimizer,
            name="checkpoint",
        )
        assert os.path.isfile(os.path.join(checkpoints_dir, "checkpoint-1.index"))

    def test_load_checkpoint(self, checkpoints_dir, basic_model, basic_optimizer):
        model = basic_model.get(fitted=True)
        new_model = basic_model.get()

        weights_model = model.get_weights()
        weights_new_model = new_model.get_weights()
        for i in range(len(weights_model)):
            assert not np.allclose(weights_model[i], weights_new_model[i])

        save_checkpoint(
            directory=checkpoints_dir,
            epoch=1,
            model=model,
            optimizer=basic_optimizer,
            name="checkpoint",
        )

        ckpt_path = os.path.join(checkpoints_dir, "checkpoint-1")
        epoch = load_checkpoint(ckpt_path, model=new_model, optimizer=basic_optimizer)
        assert epoch == 2

        weights_model = model.get_weights()
        weights_new_model = new_model.get_weights()
        for i in range(len(weights_model)):
            assert np.allclose(weights_model[i], weights_new_model[i])
