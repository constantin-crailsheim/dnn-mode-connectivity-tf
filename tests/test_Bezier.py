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
    return Bezier(degree=request.param)


@pytest.fixture(params=list(np.arange(0.0, 1.0, 0.1)))
def point_on_curve(request) -> float:
    return request.param


class TestBezier:
    @pytest.mark.parametrize("degree, expected_binom", testdata_init)
    def test_init(self, degree, expected_binom):
        curve = Bezier(degree=degree)
        assert isinstance(curve, Bezier)
        assert isinstance(curve, Curve)

        assert all(curve.binom == expected_binom)

        assert curve.range[0] == 0
        assert curve.range[-1] == degree - 1
        assert curve.range.shape == degree

        assert curve.rev_range[0] == degree - 1
        assert curve.rev_range[-1] == 0
        assert curve.rev_range.shape == degree

    @pytest.mark.parametrize("degree", [0, -1])
    def test_init_smaller_1(self, degree):
        with pytest.raises(ValueError):
            Bezier(degree=degree)

    def test_call_float(self, bezier, point_on_curve):
        output_1 = bezier(point_on_curve)
        output_2 = bezier(point_on_curve)
        output_3 = bezier(point_on_curve)
        assert np.allclose(output_1, output_2, output_3)
        assert np.allclose(sum(output_1), sum(output_2), sum(output_3), 1)
        assert output_1.shape == output_2.shape == output_3.shape == bezier.degree

    @pytest.mark.parametrize("degree, point_on_curve, expected_output", testdata_output)
    def test_call_float_specific_output(self, degree, point_on_curve, expected_output):
        bezier = Bezier(degree=degree)
        output = bezier(point_on_curve)
        assert output.shape == degree
        assert np.allclose(output, expected_output)

    def test_call_tensor(self, bezier, point_on_curve):
        output_1 = bezier(tf.constant(point_on_curve))
        output_2 = bezier(tf.constant(point_on_curve))
        output_3 = bezier(tf.constant(point_on_curve))
        assert np.allclose(output_1, output_2, output_3)
        assert np.allclose(sum(output_1), sum(output_2), sum(output_3), 1)
        assert output_1.shape == output_2.shape == output_3.shape == bezier.degree

    @pytest.mark.parametrize("degree, point_on_curve, expected_output", testdata_output)
    def test_call_tensor_specific_output(self, degree, point_on_curve, expected_output):
        bezier = Bezier(degree=degree)
        output = bezier(tf.constant(point_on_curve))
        assert output.shape == degree
        assert np.allclose(output, expected_output)

    def test_call_deterministic(self, bezier, point_on_curve):
        output1 = bezier(point_on_curve)
        output2 = bezier(point_on_curve)
        assert np.allclose(output1, output2)

    def test_call_reverseable(self, bezier, point_on_curve):
        output_regular = bezier(point_on_curve)
        output_reversed = bezier(1 - point_on_curve)
        # [::-1] -> reverse the tensor elements
        assert np.allclose(output_regular, output_reversed[::-1])
