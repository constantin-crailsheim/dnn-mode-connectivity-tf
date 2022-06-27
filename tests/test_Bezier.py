import numpy as np
import pytest
import tensorflow as tf
from mode_connectivity.curves.curves import Bezier, Curve


def list_as_tensor(list_: list, dtype=np.float32) -> tf.Tensor:
    return tf.convert_to_tensor(np.asarray(list_, dtype))


testdata_init = [
    (1, list_as_tensor([1])),
    (2, list_as_tensor([1, 1])),
    (3, list_as_tensor([1, 2, 1])),
    (4, list_as_tensor([1, 3, 3, 1])),
    (5, list_as_tensor([1, 4, 6, 4, 1])),
]

testdata_output = [
    (3, 0.0, list_as_tensor([1.0, 0.0, 0.0])),
    (3, 0.25, list_as_tensor([0.5625, 0.375, 0.0625])),
    (3, 0.5, list_as_tensor([0.25, 0.5, 0.25])),
    (3, 0.75, list_as_tensor([0.0625, 0.375, 0.5625])),
    (3, 1.0, list_as_tensor([0.0, 0.0, 1.0])),
    (5, 0.25, list_as_tensor([0.31640625, 0.421875, 0.2109375, 0.046875, 0.00390625])),
    (5, 0.75, list_as_tensor([0.00390625, 0.046875, 0.2109375, 0.421875, 0.31640625])),
]


@pytest.fixture(params=list(range(1, 5)))
def bezier(request) -> Bezier:
    return Bezier(num_bends=request.param)


@pytest.fixture(params=list(np.arange(0.0, 1.0, 0.1)))
def curve_point(request) -> float:
    return request.param


class TestBezier:
    @pytest.mark.parametrize("num_bends, expected_binom", testdata_init)
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
        assert np.allclose(sum(output_1), sum(output_2), sum(output_3), 1)
        assert output_1.shape == output_2.shape == output_3.shape == bezier.num_bends

    @pytest.mark.parametrize("num_bends, curve_point, expected_output", testdata_output)
    def test_call_float_specific_output(self, num_bends, curve_point, expected_output):
        bezier = Bezier(num_bends=num_bends)
        output = bezier(curve_point)
        assert output.shape == num_bends
        assert np.allclose(output, expected_output)

    def test_call_tensor(self, bezier, curve_point):
        output_1 = bezier(tf.constant(curve_point))
        output_2 = bezier(tf.constant(curve_point))
        output_3 = bezier(tf.constant(curve_point))
        assert np.allclose(output_1, output_2, output_3)
        assert np.allclose(sum(output_1), sum(output_2), sum(output_3), 1)
        assert output_1.shape == output_2.shape == output_3.shape == bezier.num_bends

    @pytest.mark.parametrize("num_bends, curve_point, expected_output", testdata_output)
    def test_call_tensor_specific_output(self, num_bends, curve_point, expected_output):
        bezier = Bezier(num_bends=num_bends)
        output = bezier(tf.constant(curve_point))
        assert output.shape == num_bends
        assert np.allclose(output, expected_output)

    def test_call_deterministic(self, bezier, curve_point):
        output1 = bezier(curve_point)
        output2 = bezier(curve_point)
        assert np.allclose(output1, output2)
