from this import d
import tensorflow as tf
import numpy as np

from mode_connectivity.curves.curves import Curve, Bezier, PolyChain
from mode_connectivity.curves.net import CurveNet
from mode_connectivity.curves.layers import CurveLayer
from mode_connectivity.models import CNN, MLP

import pytest


class TestCurveNet:
    testparams_description= "model,num_classes,weight_decay,curve,fix_start,fix_end,num_bends,input_shape,index"
    testparams = [
        (CNN, 10, 1e5, Bezier, True, True, 1, (128, 28, 28, 1), 0),
        (CNN, 3, 1e4, PolyChain, True, False, 3, (256, 28, 28, 1), 5-1),
        (CNN, 6, 1e3, Bezier, False, False, 5, (64, 28, 28, 1), 4),
        (MLP, 1, 1e5, Bezier, True, True, 3, (128, 32), 0),
        (MLP, 1, 1e3, PolyChain, True, False, 1, (128, 32), 3),
        (MLP, 1, 1e4, Bezier, False, False, 5, (64, 64), 7-1)
    ]

    @pytest.fixture
    def initialized_curve_net(self, request):
        model, num_classes, weight_decay, curve, fix_start, fix_end, num_bends, input_shape, index = request.param
        return CurveNet(
            num_classes,
            num_bends,
            weight_decay,
            curve,
            model.curve,
            fix_start,
            fix_end,
            model.kwargs
        )

    @pytest.fixture
    def built_curve_net(self, request, initialized_curve_net):
        model, num_classes, weight_decay, curve, fix_start, fix_end, num_bends, input_shape, index = request.param

        combined_input_shape = [
            tf.TensorShape(input_shape),
            tf.TensorShape((num_bends)),
        ]
        built_curve_net= initialized_curve_net.copy()
        built_curve_net.curve_model.build(combined_input_shape)
        return built_curve_net

    @pytest.fixture
    def base_model(self, request):
        model, num_classes, weight_decay, curve, fix_start, fix_end, num_bends, input_shape, index = request.param

        base_model= model.base(num_classes, weight_decay)
        base_model.build(input_shape= input_shape)
        return base_model

    @pytest.fixture
    def parameters(self, request):
        return request.param


    @pytest.mark.parametrize("initialized_curve_net", testparams, indirect=True)
    def test_init(self, initialized_curve_net):
        assert isinstance(initialized_curve_net.curve, Curve)
        assert isinstance(initialized_curve_net.curve_model, tf.keras.Model)
        for layer in initialized_curve_net.curve_layers:
            assert isinstance(layer, CurveLayer)

    @pytest.mark.parametrize("initialized_curve_net,parameters", [(param, param) for param in testparams], indirect=True) 
    #Repeat the parameters of each test once for each fixture that is called
    def test_call(self, initialized_curve_net, parameters):
        model, num_classes, weight_decay, curve, fix_start, fix_end, num_bends, input_shape, index = parameters
        output= initialized_curve_net(inputs= tf.random.uniform(input_shape))

        assert output.shape == (input_shape[0], num_classes)

    @pytest.mark.parametrize("initialized_curve_net,base_model,parameters", [(param, param, param) for param in testparams], indirect=True)
    def test_import_base_parameters(self, initialized_curve_net, base_model, parameters):
        model, num_classes, weight_decay, curve, fix_start, fix_end, num_bends, input_shape, index = parameters

        curve_net = initialized_curve_net
        curve_net._build_from_base_model(base_model)
        curve_weights_old = {w.name: tf.Variable(w) for w in curve_net.curve_model.variables} #tf.Variable creates a fixed copy
        curve_net.import_base_parameters(base_model, index)

        curve_weights = {w.name: w for w in curve_net.curve_model.variables}
        base_weights = {w.name: w for w in base_model.variables}

        for curve_param_name, curve_param in curve_weights.items():
            curve_param_old= curve_weights_old.get(curve_param_name)
            parameter_index= curve_net._find_parameter_index(curve_param_name)

            if parameter_index == index:
                base_param_name= curve_net._get_base_name(curve_param_name, index)
                base_param = base_weights.get(base_param_name)
                if base_param is None:
                    continue

                #Ensure that params at index are updated and not as before
                assert tf.experimental.numpy.allclose(base_param, curve_param)
                assert not tf.experimental.numpy.allclose(curve_param_old, curve_param)

            elif parameter_index != index:
                #Ensure that all other params remain as before
                assert tf.experimental.numpy.allclose(curve_param_old, curve_param)

    @pytest.mark.parametrize("initialized_curve_net,base_model,parameters", [(param, param, param) for param in testparams], indirect=True)
    def test_init_linear(self, initialized_curve_net, base_model, parameters):
        model, num_classes, weight_decay, curve, fix_start, fix_end, num_bends, input_shape, index = parameters

        curve_net = initialized_curve_net
        curve_net._build_from_base_model(base_model)
        curve_net.import_base_parameters(base_model, index)
        
        curve_net.init_linear()

        for layer in curve_net.curve_layers:
            for curve_param in ["curve_kernels", "curve_biases"]:
                params= getattr(layer, curve_param)
                first_param, last_param = params[0].value(), params[-1].value()
                min_param= tf.math.minimum(first_param, last_param)
                max_param= tf.math.maximum(first_param, last_param)

                for param in params:
                    # Check elementwise if all params lie in the range of the minimum and maximum param 
                    # of the start and end model 
                    assert tf.reduce_all(tf.math.greater_equal(param, min_param))
                    assert tf.reduce_all(tf.math.less_equal(param, max_param))