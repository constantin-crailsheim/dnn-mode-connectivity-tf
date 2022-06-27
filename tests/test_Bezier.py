import numpy as np
import pytest
import tensorflow as tf
from mode_connectivity.curves.curves import Bezier, Curve


def list_as_tensor(list_: list, dtype=np.float32) -> tf.Tensor:
    return tf.convert_to_tensor(np.asarray(list_, dtype))


testdata = [
    (1, list_as_tensor([1])),
    (2, list_as_tensor([1, 1])),
    (3, list_as_tensor([1, 2, 1])),
    (4, list_as_tensor([1, 3, 3, 1])),
    (5, list_as_tensor([1, 4, 6, 4, 1])),
]


@pytest.fixture(params=list(range(1, 5)))
def bezier(request) -> Bezier:
    return Bezier(num_bends=request.param)


@pytest.fixture(params=list(np.arange(0.0, 1.0, 0.1)))
def curve_point(request) -> float:
    return request.param


class TestBezier:
    @pytest.mark.parametrize("num_bends, expected_binom", testdata)
    def test_init(self, num_bends, expected_binom):
        curve = Bezier(num_bends=num_bends)
        assert isinstance(curve, Bezier)
        assert isinstance(curve, Curve)

        assert all(curve.binom == expected_binom)

        assert curve.range[0] == 0
        assert curve.range[-1] == num_bends - 1
        assert curve.range.shape == num_bends

        assert curve.rev_range[0] == num_bends - 1
        assert curve.rev_range[-1] == 0
        assert curve.rev_range.shape == num_bends

    @pytest.mark.parametrize("num_bends", [0, -1])
    def test_init_smaller_1(self, num_bends):
        with pytest.raises(ValueError):
            Bezier(num_bends=num_bends)

    def test_call_float(self, bezier, curve_point):
        output_1 = bezier(curve_point)
        output_2 = bezier(curve_point)
        output_3 = bezier(curve_point)
        assert np.allclose(output_1, output_2, output_3)
        assert output_1.shape == output_2.shape == output_3.shape == bezier.num_bends

    def test_call_tensor(self, bezier, curve_point):
        output_1 = bezier(tf.constant(curve_point))
        output_2 = bezier(tf.constant(curve_point))
        output_3 = bezier(tf.constant(curve_point))
        assert np.allclose(output_1, output_2, output_3)
        assert output_1.shape == output_2.shape == output_3.shape == bezier.num_bends

    def test_call_deterministic(self, bezier, curve_point):
        output1 = bezier(curve_point)
        output2 = bezier(curve_point)
        assert np.allclose(output1, output2)
