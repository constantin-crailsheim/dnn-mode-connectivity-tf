import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization

from mode_connectivity.curves import Bezier
from mode_connectivity.layers import BatchNormalizationCurve


@pytest.fixture(params=[True, False])
def fix_start(request):
    return request.param


@pytest.fixture(params=[True, False])
def fix_end(request):
    return request.param


@pytest.fixture(params=[0, 1, 2])
def fix_points(request, fix_start, fix_end):
    """Returns all possible combinations of fixed points of different lengths."""
    return [fix_start] + [False] * request.param + [fix_end]


@pytest.fixture(params=[None, tf.keras.regularizers.L2(0.5)])
def regularizer(request):
    return request.param


gamma_regularizer = regularizer
beta_regularizer = regularizer


@pytest.fixture(params=[2, 8, 32])
def single_input_shape(request):
    return [request.param]


@pytest.fixture(params=[1, 2, 3])
def input_shape(request, single_input_shape):
    return single_input_shape * request.param


@pytest.fixture()
def random_input(input_shape):
    return tf.random.uniform(input_shape)


@pytest.fixture
def batch_norm_layer(fix_points, gamma_regularizer, beta_regularizer):
    return BatchNormalizationCurve(
        fix_points=fix_points,
        gamma_regularizer=gamma_regularizer,
        beta_regularizer=beta_regularizer,
    )


@pytest.fixture
def batch_norm_built(batch_norm_layer, input_shape):
    # Add curve weights shape to input shape
    full_input_shape = (input_shape, (len(batch_norm_layer.fix_points),))
    batch_norm_layer.build(full_input_shape)
    return batch_norm_layer


class TestBatchNormalization:
    def test_init(self, batch_norm_layer):
        assert batch_norm_layer.num_bends == len(batch_norm_layer.fix_points) - 2
        assert batch_norm_layer.parameter_types == ("gamma", "beta")
        assert batch_norm_layer.base_layer == tf.keras.layers.BatchNormalization

    def test_build_attributes(self, batch_norm_built):
        assert not hasattr(batch_norm_built, "gamma")
        assert not hasattr(batch_norm_built, "beta")

        assert hasattr(batch_norm_built, "curve_gammas")
        assert hasattr(batch_norm_built, "curve_betas")
        assert hasattr(batch_norm_built, "moving_mean")
        assert hasattr(batch_norm_built, "moving_variance")

        n_fix_points = len(batch_norm_built.fix_points)
        param_names = [v.name for v in batch_norm_built.variables]
        assert len(param_names) == n_fix_points * 2 + 2
        assert "moving_mean:0" in param_names
        assert "moving_variance:0" in param_names
        for i in range(n_fix_points):
            assert f"gamma_curve_{i}:0" in param_names
            assert f"beta_curve_{i}:0" in param_names

    def test_call_params_updated(self, batch_norm_layer, random_input):
        curve_weights = Bezier(batch_norm_layer.num_bends)(0.5)
        batch_norm_layer((random_input, curve_weights))
        assert hasattr(batch_norm_layer, "gamma")
        assert hasattr(batch_norm_layer, "beta")

    def test_call(self, fix_points, input_shape):
        bn_default = BatchNormalization()
        bn_curve = BatchNormalizationCurve(fix_points=fix_points)

        inputs = tf.random.uniform(input_shape)
        curve_weights = [1] + [0] * (len(bn_curve.fix_points) - 1)
        curve_weights = tf.convert_to_tensor(np.asarray(curve_weights, np.float32))
        output_default = bn_default(inputs)
        output_curve = bn_curve((inputs, curve_weights))
        assert np.allclose(output_default, output_curve)

    def test_call_moving(self, fix_points, input_shape):
        bn_curve = BatchNormalizationCurve(fix_points=fix_points)

        inputs = tf.random.uniform(input_shape)
        curve_weights = [1] + [0] * (len(bn_curve.fix_points) - 1)
        curve_weights = tf.convert_to_tensor(np.asarray(curve_weights, np.float32))

        bn_curve.build((input_shape, curve_weights.shape))

        expected_mean = tf.zeros(input_shape[0])
        expected_variance = tf.ones(input_shape[0])
        assert np.allclose(bn_curve.moving_mean, expected_mean)
        assert np.allclose(bn_curve.moving_variance, expected_variance)

        # Call BatchNorm and check if moving stats are updated
        bn_curve((inputs, curve_weights))
        assert np.allclose(bn_curve.moving_mean, expected_mean)
        assert np.allclose(bn_curve.moving_variance, expected_variance)

        # Training=False should not impact stats update
        bn_curve((inputs, curve_weights), training=False)
        assert np.allclose(bn_curve.moving_mean, expected_mean)
        assert np.allclose(bn_curve.moving_variance, expected_variance)

        # Training=True should
        bn_curve((inputs, curve_weights), training=True)
        assert not np.allclose(bn_curve.moving_mean, expected_mean)
        assert not np.allclose(bn_curve.moving_variance, expected_variance)
