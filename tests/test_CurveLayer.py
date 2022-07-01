import tensorflow as tf
import numpy as np

from mode_connectivity.curves.layers import CurveLayer, Conv2DCurve, DenseCurve
from mode_connectivity.curves.curves import Bezier

from abc import abstractmethod
import pytest


class CurveLayerTest:
    fix_points = [True, False, True]

    def curve_point_weights(self):
        num_bends= len(self.fix_points)
        bezier_curve= Bezier(num_bends)
        rand_t= tf.random.uniform(shape=(1,))
        return bezier_curve(rand_t)

    @pytest.fixture(name="curve_point_weights")
    def curve_point_weights_fixture(self):
        # Implemented as a directly callable fixture in order to ensure update of curve_point_weights
        # https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly
        return self.curve_point_weights()

    @abstractmethod
    @pytest.fixture
    def initialized_layer(self, *args, **kwargs):
        pass

    @pytest.fixture
    def built_layer(self, initialized_layer, curve_point_weights):
        output= initialized_layer([tf.random.uniform(shape=self.input_shape), curve_point_weights])
        built_layer= initialized_layer
        return built_layer


    @abstractmethod
    def test_init(self, *args, **kwargs):
        pass

    def test_build(self, built_layer):
        # Build in parent class tf.keras.layers... should add kernel and bias parameters 
        # (Even though they won't be used directly)        
        assert built_layer.kernel != None
        assert built_layer.bias != None

        # Conv2DCurve build() should add curve_kernels and curve_biases parameters    
        assert built_layer.curve_kernels != None
        assert built_layer.curve_biases != None

        for (param, curve_param) in [("kernel" ,"curve_kernels"), ("bias", "curve_biases")]:
            temp_param= getattr(built_layer, curve_param)
            temp_curve_param= getattr(built_layer, param)

            assert len(temp_param) == len(self.fix_points)
            for i in range(len(self.fix_points)): #Fix points
                assert temp_param[i].shape == temp_curve_param.shape
                if param != "bias":
                    assert tf.math.count_nonzero(temp_param[i]) != 0 #Check if initialized

    def test_call(self, built_layer):
        #curve_point_weights() has to be called directly in order to ensure that the curve_point_weights are different from the ones used to build the model.
        curve_point_weights= self.curve_point_weights() 

        # Even without optimization of the networks parameters, the params should be updated since we change the location on the curve.
        old_kernel= tf.Variable(built_layer.kernel)
        old_bias= tf.Variable(built_layer.bias)
        output= built_layer([tf.random.uniform(shape=self.input_shape), curve_point_weights])
        new_kernel= tf.Variable(built_layer.kernel)
        new_bias= tf.Variable(built_layer.bias)

        assert not tf.experimental.numpy.allclose(old_kernel, new_kernel)
        assert not tf.experimental.numpy.allclose(old_bias, new_bias)

        self.check_output_size(output)

    @abstractmethod
    def check_output_size(self, output):
        pass


    def test_compute_weighted_parameters(self, built_layer, curve_point_weights):
        #Call() calls compute_weighted_parameters()
        output= built_layer([tf.random.uniform(shape=self.input_shape), curve_point_weights])

        #Check if alternative calculation of weighted params leads to the same kernels and biases
        for param_type, curve_param_type in [("kernel", "curve_kernels"), ("bias", "curve_biases")]:
            alt_param=None
            for i in range(len(getattr(built_layer, curve_param_type))):
                scaled_param= curve_point_weights[i] * getattr(built_layer, curve_param_type)[i]
                if alt_param == None:
                    alt_param = scaled_param
                else:
                    alt_param += scaled_param

            assert tf.experimental.numpy.allclose(getattr(built_layer, param_type), alt_param)

    # Still necessary if we have test_init?
    # def test_init_direct(self): 
    #     with pytest.raises(
    #         TypeError,
    #         match="Can't instantiate abstract class CurveLayer with abstract methods build, call",
    #     ):
    #         CurveLayer([True, False, True])


class TestConv2DCurveLayer(CurveLayerTest):
    filters = 32
    kernel_size=(3, 3)
    input_shape= (128, 28, 28, 1)

    @pytest.fixture
    def initialized_layer(self):
        return Conv2DCurve(self.filters, self.kernel_size, self.fix_points)

    def test_init(self, initialized_layer):    
        assert isinstance(initialized_layer, CurveLayer)
        assert isinstance(initialized_layer, tf.keras.layers.Conv2D)

    def check_output_size(self, output):
        output_shape_h = self.input_shape[1] - self.kernel_size[0] + 1
        output_shape_w = self.input_shape[2] - self.kernel_size[1] + 1
        assert output.shape == (self.input_shape[0], output_shape_h, output_shape_w, self.filters)


class TestDenseCurveLayer(CurveLayerTest):
    units = 16
    input_shape= (128, 32)  

    @pytest.fixture
    def initialized_layer(self):
        return DenseCurve(self.units, self.fix_points)

    def test_init(self, initialized_layer):
        assert isinstance(initialized_layer, CurveLayer)
        assert isinstance(initialized_layer, tf.keras.layers.Dense)   

    def check_output_size(self, output):
        assert output.shape == (self.input_shape[0], self.units)