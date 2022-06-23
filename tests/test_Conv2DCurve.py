import tensorflow as tf
from mode_connectivity.curves.curves import Bezier
from mode_connectivity.curves.layers import Conv2DCurve, CurveLayer


class TestConv2DCurve:
    def test_inheritance(self):
        assert issubclass(Conv2DCurve, CurveLayer)
        assert issubclass(Conv2DCurve, tf.keras.layers.Conv2D)

    def test_init(self):
        layer = Conv2DCurve(
            filters=32, kernel_size=(3, 3), fix_points=[True, False, True]
        )
        assert isinstance(layer, CurveLayer)
        assert isinstance(layer, tf.keras.layers.Conv2D)
        layer.build((28, 28))

    def test_call(self):
        layer = Conv2DCurve(
            filters=32, kernel_size=(3, 3), fix_points=[True, False, True]
        )
        inputs = tf.random.uniform(shape=(128, 28, 28, 1))
        curve = Bezier(3)
        coeffs_t = curve(0.5)
        layer(inputs, coeffs_t)  # layer.call()
