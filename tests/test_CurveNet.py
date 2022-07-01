import tensorflow as tf
import numpy as np

from mode_connectivity.curves.curves import Curve, Bezier
from mode_connectivity.curves.net import CurveNet
from mode_connectivity.curves.layers import CurveLayer
from mode_connectivity.models import CNN, MLP

import pytest

from mode_connectivity.models.cnn import CNNCurve


class TestCurveNet:
    @pytest.fixture
    def model_dir(tmpdir):
        return tmpdir.mkdir("model")

    @pytest.fixture
    def initialized_curve_net(self):
        return CurveNet(
            num_classes=10,
            num_bends=3,
            weight_decay=1e4,
            curve=Bezier,
            curve_model=CNN.curve,
            fix_start=True,
            fix_end=True,
            architecture_kwargs=CNN.kwargs,
        )
    # Same for MLP.curve

    @pytest.fixture
    def built_curve_net(self, initialized_curve_net):
        input_shape = [
            tf.TensorShape((128, 28, 28, 1)),
            tf.TensorShape((3)),
        ]
        initialized_curve_net.curve_model.build(input_shape)
        built_curve_net= initialized_curve_net
        return built_curve_net

    # Same for MLP.curve

    def test_init(self, initialized_curve_net):
        assert isinstance(initialized_curve_net.curve, Curve)
        assert isinstance(initialized_curve_net.curve_model, tf.keras.Model)
        for layer in initialized_curve_net.curve_layers:
            assert isinstance(layer, CurveLayer)

    def test_call(self, initialized_curve_net):
        output= initialized_curve_net(inputs= tf.random.uniform((128, 28, 28, 1)))

        assert output.shape == (128, 10) #(input_size[0], num_classes)

    def test_import_base_parameters(self, initialized_curve_net):
       #TO DO: Als Fixture mit anderen Parametrisierungen, zb index letzter

        curve_net = initialized_curve_net
        base_model= CNN.base(10, 1e4)
        base_model.build(input_shape=(None, 28, 28, 1))
        curve_net._build_from_base_model(base_model)
        curve_weights_old = {w.name: tf.Variable(w) for w in curve_net.curve_model.variables} #tf.Variable creates a fixed copy
        index= 0
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


    def test_init_linear(self, initialized_curve_net):
        curve_net = initialized_curve_net
        base_model= CNN.base(10, 1e4)
        base_model.build(input_shape=(None, 28, 28, 1))
        curve_net._build_from_base_model(base_model)
        index= 0
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