import tensorflow as tf
import numpy as np

from mode_connectivity.curves.layers import CurveLayer, Conv2DCurve, DenseCurve
from mode_connectivity.curves.curves import Bezier

from abc import abstractmethod
import pytest


class CurveLayerTest:
    testparams_description= "filters,units,kernel_size,input_shape,fix_points"

    def curve_point_weights(self, fix_points):
        num_bends= len(fix_points)
        bezier_curve= Bezier(num_bends)
        rand_t= tf.random.uniform(shape=(1,))
        return bezier_curve(rand_t)

    @pytest.fixture(name="curve_point_weights")
    def curve_point_weights_fixture(self, request):
        # Implemented as a directly callable fixture in order to ensure update of curve_point_weights
        # https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly
        try:
            filters, units, kernel_size, input_shape, fix_points= request.param
        except:
            filters, units, kernel_size, input_shape, fix_points= request._parent_request.param

        return self.curve_point_weights(fix_points)

    @abstractmethod
    @pytest.fixture
    def initialized_layer(self, *args, **kwargs):
        pass

    @pytest.fixture
    def built_layer(self, request, initialized_layer, curve_point_weights):
        filters, units, kernel_size, input_shape, fix_points = request.param
        output= initialized_layer([tf.random.uniform(shape=input_shape), curve_point_weights])
        built_layer= initialized_layer
        return built_layer

    @pytest.fixture
    def parameters(self, request):
        return request.param


    @abstractmethod
    def test_init(self, *args, **kwargs):
        pass

    def test_build(self, built_layer, parameters):
        filters, units, kernel_size, input_shape, fix_points = parameters

        # Build in parent class tf.keras.layers... should add kernel and bias parameters 
        # (Even though they won't be used directly)        
        assert built_layer.kernel != None
        assert built_layer.bias != None

        # Conv2DCurve build() should add curve_kernels and curve_biases parameters    
        assert built_layer.curve_kernels != None
        assert built_layer.curve_biases != None

        for (param, curve_param) in [("kernel" ,"curve_kernels"), ("bias", "curve_biases")]:
            temp_param= getattr(built_layer, param)
            temp_curve_param= getattr(built_layer, curve_param)

            assert len(temp_curve_param) == len(fix_points)
            for i in range(len(fix_points)):
                #Ensure that curve kernels and biases are of the same shape as of the original net
                assert temp_curve_param[i].shape == temp_param.shape
                if param != "bias":
                    #Check if initialized
                    assert tf.math.count_nonzero(temp_curve_param[i]) != 0 

    def test_call(self, built_layer, parameters):
        filters, units, kernel_size, input_shape, fix_points = parameters

        #curve_point_weights() has to be called directly in order to ensure that the curve_point_weights are different from the ones used to build the model.
        curve_point_weights= self.curve_point_weights(fix_points) 

        # Even without optimization of the networks parameters, the params should be updated since we change the location on the curve.
        old_kernel= tf.Variable(built_layer.kernel)
        old_bias= tf.Variable(built_layer.bias)
        output= built_layer([tf.random.uniform(shape=input_shape), curve_point_weights])
        new_kernel= tf.Variable(built_layer.kernel)
        new_bias= tf.Variable(built_layer.bias)

        assert not tf.experimental.numpy.allclose(old_kernel, new_kernel)
        assert not tf.experimental.numpy.allclose(old_bias, new_bias)

        self.check_output_size(output, parameters)

    @abstractmethod
    def check_output_size(self, output):
        pass

    def test_compute_weighted_parameters(self, built_layer, curve_point_weights, parameters):
        filters, units, kernel_size, input_shape, fix_points = parameters

        #Call() calls compute_weighted_parameters()
        output= built_layer([tf.random.uniform(shape=input_shape), curve_point_weights])

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