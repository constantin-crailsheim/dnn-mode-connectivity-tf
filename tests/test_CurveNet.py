import tensorflow as tf

from mode_connectivity.curves import Curve, Bezier, PolyChain
from mode_connectivity.net import CurveNet
from mode_connectivity.layers import CurveLayer
from showcase.models import CNN, MLP

import pytest

class TestCurveNet:
    testparams_description = "model,num_classes,weight_decay,curve,fix_start,fix_end,num_bends,input_shape,index"
    testparams = [
        (CNN, 10, 1e5, Bezier, True, True, 1, (128, 28, 28, 1), 0),
        (CNN, 3, 1e4, PolyChain, True, False, 3, (256, 28, 28, 1), 5 - 1),
        (CNN, 6, 1e3, Bezier, False, False, 5, (64, 28, 28, 1), 4),
        (MLP, 1, 1e5, Bezier, True, True, 3, (128, 32), 0),
        (MLP, 1, 1e3, PolyChain, True, False, 1, (128, 32), 0),
        (MLP, 1, 1e4, Bezier, False, False, 5, (64, 64), 7 - 1),
    ]


    @pytest.fixture
    def parameters(self, request):
        return request.param


    @pytest.mark.parametrize(
        "parameters", 
        [(param) for param in testparams], 
        indirect=True)
    def test_import_base_parameters(self, parameters):
        (
            model, 
            num_classes, 
            weight_decay, 
            curve, 
            fix_start, 
            fix_end, 
            num_bends, 
            input_shape, 
            index
         ) = parameters

        base_model= model.base(num_classes, weight_decay)
        base_model.build(input_shape= input_shape)

        curve_net= CurveNet(
            num_classes,
            num_bends,
            weight_decay,
            curve,
            model.curve,
            fix_start,
            fix_end,
            model.kwargs,
        )

        curve_net._build_from_base_model(base_model)        

        curve_weights_old = {
            w.name: tf.Variable(w) for w in curve_net.curve_model.variables
            } #tf.Variable creates a fixed copy

        curve_net.import_base_parameters(base_model, index)

        curve_weights = {
            w.name: w for w in curve_net.curve_model.variables
            }
        base_weights = {
            w.name: w for w in base_model.variables
            }

        for curve_param_name, curve_param in curve_weights.items():
            curve_param_old = curve_weights_old.get(curve_param_name)
            parameter_index = curve_net._find_parameter_index(curve_param_name)

            if parameter_index == index:
                base_param_name = curve_net._get_base_name(curve_param_name, index)
                base_param = base_weights.get(base_param_name)
                if base_param is None:
                    continue

                # Ensure that params at index are updated and not as before
                assert tf.experimental.numpy.allclose(base_param, curve_param)
                if "kernel" in curve_param_name:
                    assert not tf.experimental.numpy.allclose(curve_param_old, curve_param)

            elif parameter_index != index:
                # Ensure that all other params remain as before
                assert tf.experimental.numpy.allclose(curve_param_old, curve_param)

        del base_model, curve_weights_old, curve_weights, base_weights, curve_net, model, num_classes, weight_decay, curve, fix_start, fix_end, num_bends, input_shape, index, parameters


    @pytest.mark.parametrize(
        "parameters", 
        [(param) for param in testparams], 
        indirect=True)
    def test_init_linear(self, parameters):
        (
            model, 
            num_classes, 
            weight_decay, 
            curve, 
            fix_start, 
            fix_end, 
            num_bends, 
            input_shape, 
            index
         ) = parameters

        base_model= model.base(num_classes, weight_decay)
        base_model.build(input_shape= input_shape)

        curve_net= CurveNet(
            num_classes,
            num_bends,
            weight_decay,
            curve,
            model.curve,
            fix_start,
            fix_end,
            model.kwargs
        )
        curve_net._build_from_base_model(base_model)   

        curve_net.import_base_parameters(base_model, index)

        curve_net.init_linear()

        for layer in curve_net.curve_layers:
            curve_params = [
                CurveLayer._get_curve_param_name(p) for p in layer.parameters
            ]
            for curve_param in curve_params:
                params = getattr(layer, curve_param)
                first_param, last_param = params[0].value(), params[-1].value()
                min_param = tf.math.minimum(first_param, last_param)
                max_param = tf.math.maximum(first_param, last_param)

                for param in params:
                    # Check elementwise if all params lie in the range of the minimum and maximum param
                    # of the start and end model
                    assert tf.reduce_all(tf.math.greater_equal(param, min_param))
                    assert tf.reduce_all(tf.math.less_equal(param, max_param))

        del base_model, curve_net, model, num_classes, weight_decay, curve, fix_start, fix_end, num_bends, input_shape, index, parameters


    @pytest.mark.parametrize("parameters", [(param) for param in testparams], indirect=True)
    def test_init(self, parameters):
        (
            model, 
            num_classes, 
            weight_decay, 
            curve, 
            fix_start, 
            fix_end, 
            num_bends, 
            input_shape, 
            index
         ) = parameters

        curve_net= CurveNet(
            num_classes,
            num_bends,
            weight_decay,
            curve,
            model.curve,
            fix_start,
            fix_end,
            model.kwargs
        )

        assert isinstance(curve_net.curve, Curve)
        assert isinstance(curve_net.curve_model, tf.keras.Model)
        for layer in curve_net.curve_layers:
            assert isinstance(layer, CurveLayer)

        del curve_net, model, num_classes, weight_decay, curve, fix_start, fix_end, num_bends, input_shape, index, parameters


    @pytest.mark.parametrize("parameters", [(param) for param in testparams], indirect=True)
    def test_call(self, parameters):
        (
            model, 
            num_classes, 
            weight_decay, 
            curve, 
            fix_start, 
            fix_end, 
            num_bends, 
            input_shape, 
            index
         ) = parameters

        curve_net= CurveNet(
            num_classes,
            num_bends,
            weight_decay,
            curve,
            model.curve,
            fix_start,
            fix_end,
            model.kwargs
        )
        output= curve_net(inputs= tf.random.uniform(input_shape))

        assert output.shape == (input_shape[0], num_classes)

        del curve_net, model, num_classes, weight_decay, curve, fix_start, fix_end, num_bends, input_shape, index, parameters, output