import numpy as np
import pytest
import tensorflow as tf
from mode_connectivity.curves.layers import CurveLayer


class TestCurveLayer:
    def test_init_direct(self):
        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class CurveLayer with abstract methods build, call",
        ):
            CurveLayer([True, False, True])

    def test_compute_weights(self):
        shape = (3, 3, 1, 32)
        num_bends = 3
        weights = [tf.random.uniform(shape) for _ in range(num_bends)]
        curve_point_weights = tf.random.uniform([num_bends])

        # New method (matrix multiplication)
        combined_weights = tf.stack([w for w in weights], axis=-1)
        weights_avg_new = tf.linalg.matvec(combined_weights, curve_point_weights)

        assert weights_avg_new.shape == shape

        # Old method (Looping)
        weights_avg_old = 0
        for i in range(curve_point_weights.shape[0]):
            weights_avg_old += weights[i] * curve_point_weights[i]

        assert weights_avg_old.shape == shape

        # Assert both give equal results
        assert np.allclose(weights_avg_new, weights_avg_old)
