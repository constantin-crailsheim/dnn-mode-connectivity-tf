
from tests import CurveLayerTest

import tensorflow as tf
import numpy as np

from mode_connectivity.curves.layers import CurveLayer, Conv2DCurve, DenseCurve
from mode_connectivity.curves.curves import Bezier

from abc import abstractmethod
import pytest

class TestConv2DCurveLayer(CurveLayerTest):
    testparams = [
        (32, None, (3, 3), (128, 28, 28, 1), [True, True, True]),
        (64, None, (3, 5), (256, 28, 28, 1), [True, False, True]),
        (16, None, (1, 1), (256, 28, 28, 1), [False, False, False])
    ]
    #Conv2D does not have the parameter "units". Hence it is set to None.

    @pytest.fixture
    def initialized_layer(self, request):
        try:
            filters, units, kernel_size, input_shape, fix_points= request.param
        except:
            filters, units, kernel_size, input_shape, fix_points= request._parent_request.param
        return Conv2DCurve(filters, kernel_size, fix_points)

    @pytest.mark.parametrize("initialized_layer", testparams, indirect=True)
    def test_init(self, initialized_layer):    
        assert isinstance(initialized_layer, CurveLayer)
        assert isinstance(initialized_layer, tf.keras.layers.Conv2D)

    @pytest.mark.parametrize("built_layer,parameters", [(param, param) for param in testparams], indirect=True) 
    #Repeat the parameters of each test once for each fixture that is called
    def test_build(self, built_layer, parameters):
        super().test_build(built_layer, parameters)

    @pytest.mark.parametrize("built_layer,parameters", [(param, param) for param in testparams], indirect=True) 
    def test_call(self, built_layer, parameters):
        super().test_call(built_layer, parameters)

    def check_output_size(self, output, parameters):
        filters, units, kernel_size, input_shape, fix_points = parameters
        output_shape_h = input_shape[1] - kernel_size[0] + 1
        output_shape_w = input_shape[2] - kernel_size[1] + 1
        assert output.shape == (input_shape[0], output_shape_h, output_shape_w, filters)

    @pytest.mark.parametrize("built_layer,curve_point_weights,parameters", [(param, param, param) for param in testparams], indirect=True) 
    def test_compute_weighted_parameters(self, built_layer, curve_point_weights, parameters):
        super().test_compute_weighted_parameters(built_layer, curve_point_weights, parameters)