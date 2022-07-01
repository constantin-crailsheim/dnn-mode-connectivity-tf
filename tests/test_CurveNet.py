import numpy as np
import pytest
import tensorflow as tf
from mode_connectivity.curves.curves import Bezier
from mode_connectivity.curves.net import CurveNet
from mode_connectivity.models import CNN


@pytest.fixture
def model_dir(tmpdir):
    return tmpdir.mkdir("model")


@pytest.fixture(params=list(range(0, 4)))
def num_bends(request) -> int:
    return request.param


class TestCurveNet:
    def test_init(self, num_bends):
        net = CurveNet(
            num_classes=10,
            num_bends=num_bends,
            weight_decay=1e4,
            curve=Bezier,
            curve_model=CNN.curve,
            fix_start=True,
            fix_end=True,
            architecture_kwargs=CNN.kwargs,
        )

    def test_call(self):
        net = CurveNet(
            num_classes=10,
            num_bends=0,
            weight_decay=1e4,
            curve=Bezier,
            curve_model=CNN.curve,
            fix_start=True,
            fix_end=True,
            architecture_kwargs=CNN.kwargs,
        )
        net(tf.random.uniform((128, 28, 28, 1)))

    def test_save_model(self, model_dir, num_bends):
        net = CurveNet(
            num_classes=10,
            num_bends=num_bends,
            weight_decay=1e4,
            curve=Bezier,
            curve_model=CNN.curve,
            fix_start=True,
            fix_end=True,
            architecture_kwargs=CNN.kwargs,
        )
        net(tf.random.uniform((128, 28, 28, 1)))
        net.save(filepath=model_dir)

    def test_load_model_and_call(self, model_dir):
        net = CurveNet(
            num_classes=10,
            num_bends=3,
            weight_decay=1e4,
            curve=Bezier,
            curve_model=CNN.curve,
            fix_start=True,
            fix_end=True,
            architecture_kwargs=CNN.kwargs,
        )
        net(tf.random.uniform((128, 28, 28, 1)))
        net.compile()
        net.save(filepath=model_dir)

        net2 = tf.keras.models.load_model(filepath=model_dir)
        x = tf.random.uniform((10, 28, 28, 1))
        assert np.allclose(net.predict(x), net2.predict(x))

        net2(tf.random.uniform((128, 28, 28, 1)))
        net2.point_on_curve = net2.generate_point_on_curve()
        net2(tf.random.uniform((128, 28, 28, 1)), training=False)

        for layer_net, layer_net2 in zip(net.curve_layers, net2.curve_layers):
            assert all(
                np.allclose(k1.value(), k2.value())
                for k1, k2 in zip(layer_net.curve_kernels, layer_net2.curve_kernels)
            )
            assert all(
                np.allclose(b1.value(), b2.value())
                for b1, b2 in zip(layer_net.curve_biases, layer_net2.curve_biases)
            )

    def test_save_weights(self, model_dir, num_bends):
        net = CurveNet(
            num_classes=10,
            num_bends=num_bends,
            weight_decay=1e4,
            curve=Bezier,
            curve_model=CNN.curve,
            fix_start=True,
            fix_end=True,
            architecture_kwargs=CNN.kwargs,
        )
        net(tf.random.uniform((128, 28, 28, 1)))
        net.save_weights(filepath=model_dir)

    def test_load_weights(self, model_dir, num_bends):
        net = CurveNet(
            num_classes=10,
            num_bends=num_bends,
            weight_decay=1e4,
            curve=Bezier,
            curve_model=CNN.curve,
            fix_start=True,
            fix_end=True,
            architecture_kwargs=CNN.kwargs,
        )
        net(tf.random.uniform((128, 28, 28, 1)))
        net(tf.random.uniform((128, 28, 28, 1)))
        net(tf.random.uniform((128, 28, 28, 1)))
        net.save_weights(filepath=model_dir)

        net2 = CurveNet(
            num_classes=10,
            num_bends=num_bends,
            weight_decay=1e4,
            curve=Bezier,
            curve_model=CNN.curve,
            fix_start=True,
            fix_end=True,
            architecture_kwargs=CNN.kwargs,
        )
        net2.build(input_shape=tf.TensorShape((None, 28, 28, 1)))

        net_weights = net.get_weights()
        net2_weights = net2.get_weights()
        assert not all(np.allclose(w1, w2) for w1, w2 in zip(net_weights, net2_weights))

        net2.load_weights(filepath=model_dir)

        net_weights = net.get_weights()
        net2_weights = net2.get_weights()
        assert all(np.allclose(w1, w2) for w1, w2 in zip(net_weights, net2_weights))
