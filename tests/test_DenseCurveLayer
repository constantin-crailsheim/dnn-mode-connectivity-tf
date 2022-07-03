from tests import CurveLayerTest

import tensorflow as tf
import numpy as np

from mode_connectivity.curves.layers import CurveLayer, Conv2DCurve, DenseCurve
from mode_connectivity.curves.curves import Bezier

from abc import abstractmethod
import pytest

class TestDenseCurveLayer(CurveLayerTest):
    testparams = [
        (None, 16, None, (128, 32), [True, True, True]),
        (None, 32, None, (128, 64), [True, False, True]),
        (None, 8, None, (128, 16), [False, False, False])
    ]
    #Conv2D does not have the parameters "filters" and "kernel_size". Hence they are set to None.

    @pytest.fixture
    def initialized_layer(self, request):
        try:
            filters, units, kernel_size, input_shape, fix_points= request.param
        except:
            filters, units, kernel_size, input_shape, fix_points= request._parent_request.param

        return DenseCurve(units, fix_points)

    @pytest.mark.parametrize("initialized_layer", testparams, indirect=True)
    def test_init(self, initialized_layer):
        assert isinstance(initialized_layer, CurveLayer)
        assert isinstance(initialized_layer, tf.keras.layers.Dense)   

    @pytest.mark.parametrize("built_layer,parameters", [(param, param) for param in testparams], indirect=True) 
    #Repeat the parameters of each test once for each fixture that is called
    def test_build(self, built_layer, parameters):
        super().test_build(built_layer, parameters)

    @pytest.mark.parametrize("built_layer,parameters", [(param, param) for param in testparams], indirect=True) 
    def test_call(self, built_layer, parameters):
        super().test_call(built_layer, parameters)

    def check_output_size(self, output, parameters):
        filters, units, kernel_size, input_shape, fix_points = parameters
        assert output.shape == (input_shape[0], units)

    @pytest.mark.parametrize("built_layer,curve_point_weights,parameters", [(param, param, param) for param in testparams], indirect=True) 
    def test_compute_weighted_parameters(self, built_layer, curve_point_weights, parameters):
        super().test_compute_weighted_parameters(built_layer, curve_point_weights, parameters)