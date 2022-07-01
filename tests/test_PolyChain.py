import numpy as np
import pytest
import tensorflow as tf
from mode_connectivity.curves.curves import PolyChain, Curve


def list_as_tensor(list_: list, dtype=np.float32) -> tf.Tensor:
    return tf.convert_to_tensor(np.asarray(list_, dtype))

testdata_output = [
    (1, 0.0, list_as_tensor([1.0, 0.0, 0.0])),
    (1, 0.25, list_as_tensor([0.5, 0.5, 0.0])),
    (1, 0.5, list_as_tensor([0.0, 1.0, 0.0])),
    (1, 0.75, list_as_tensor([0.0, 0.5, 0.5])),
    (1, 1.0, list_as_tensor([0.0, 0.0, 1.0])),
    (3, 0.25, list_as_tensor([0.0, 1.0, 0.0, 0.0, 0.0])),
    (3, 0.75, list_as_tensor([0.0, 0.0, 0.0, 1.0, 0.0])),
]


@pytest.fixture(params=list(range(1, 5)))
def polychain(request) -> PolyChain:
    return PolyChain(num_bends=request.param)


@pytest.fixture(params=list(np.arange(0.0, 1.0, 0.1)))
def point_on_curve(request) -> float:
    return request.param


class TestPolyChain:
    @pytest.mark.parametrize("num_bends", [1, 2, 3, 4, 5])
    def test_init(self, num_bends):
        curve = PolyChain(num_bends=num_bends)
        assert isinstance(curve, PolyChain)
        assert isinstance(curve, Curve)

        assert curve.range[0] == 0
        assert curve.range[-1] == num_bends + 1
        assert curve.range.shape == num_bends + 2

    @pytest.mark.parametrize("num_bends", [0, -1])
    def test_init_smaller_1(self, num_bends):
        with pytest.raises(ValueError):
            PolyChain(num_bends=num_bends)

    def test_call_float(self, polychain, point_on_curve):
        output_1 = polychain(point_on_curve)
        output_2 = polychain(point_on_curve)
        output_3 = polychain(point_on_curve)
        assert np.allclose(output_1, output_2, output_3)
        assert np.allclose(sum(output_1), sum(output_2), sum(output_3), 1)
        assert output_1.shape == output_2.shape == output_3.shape == polychain.num_bends + 2

    @pytest.mark.parametrize(
        "num_bends, point_on_curve, expected_output", testdata_output
    )
    def test_call_float_specific_output(
        self, num_bends, point_on_curve, expected_output
    ):
        polychain = PolyChain(num_bends=num_bends)
        output = polychain(point_on_curve)
        assert output.shape == num_bends + 2
        assert np.allclose(output, expected_output)

    def test_call_tensor(self, polychain, point_on_curve):
        output_1 = polychain(tf.constant(point_on_curve))
        output_2 = polychain(tf.constant(point_on_curve))
        output_3 = polychain(tf.constant(point_on_curve))
        assert np.allclose(output_1, output_2, output_3)
        assert np.allclose(sum(output_1), sum(output_2), sum(output_3), 1)
        assert output_1.shape == output_2.shape == output_3.shape == polychain.num_bends + 2

    @pytest.mark.parametrize(
        "num_bends, point_on_curve, expected_output", testdata_output
    )
    def test_call_tensor_specific_output(
        self, num_bends, point_on_curve, expected_output
    ):
        polychain = PolyChain(num_bends=num_bends)
        output = polychain(tf.constant(point_on_curve))
        assert output.shape == num_bends + 2
        assert np.allclose(output, expected_output)

    def test_call_deterministic(self, polychain, point_on_curve):
        output1 = polychain(point_on_curve)
        output2 = polychain(point_on_curve)
        assert np.allclose(output1, output2)

    def test_call_reverseable(self, polychain, point_on_curve):
        output_regular = polychain(point_on_curve)
        output_reversed = polychain(1 - point_on_curve)
        # [::-1] -> reverse the tensor elements
        assert np.allclose(output_regular, output_reversed[::-1])
