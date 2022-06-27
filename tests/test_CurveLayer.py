import tensorflow as tf
import numpy as np

from mode_connectivity.curves.layers import CurveLayer, Conv2DCurve, DenseCurve
from mode_connectivity.curves.curves import Bezier

from abc import abstractmethod
import pytest


class CurveLayerTest:
    fix_points = [True, False, True]

    @abstractmethod
    def init_standard_layer(self, *args, **kwargs):
        pass

    @abstractmethod
    def call_standard_layer(self, *args, **kwargs):
        pass

    def get_curve_point_weights(self):
        num_bends= len(self.fix_points)
        bezier_curve= Bezier(num_bends)
        rand_t= tf.random.uniform(shape=(1,))
        return bezier_curve(rand_t)


    @abstractmethod
    def test_init(self, *args, **kwargs):
        pass


    def test_build(self):
        layer = self.init_standard_layer()
        #Also calls build()
        self.call_standard_layer(layer)
        
        self.built_layer_test(layer) 

    def built_layer_test(self, built_layer):
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

    def test_call(self):
        layer = self.init_standard_layer()
        self.param_updates_test(layer)

        output, curve_point_weights= self.call_standard_layer(layer)
        self.check_output_size(output)

    def param_updates_test(self, layer):
        # Even without optimization of the network parameters, the params should be updated
        # since we change the location on the curve/ randomly set curve_point_weights of the submodels.
        self.call_standard_layer(layer)
        old_kernel= layer.kernel
        old_bias= layer.bias
        self.call_standard_layer(layer)
        new_kernel= layer.kernel
        new_bias= layer.bias

        assert not tf.experimental.numpy.allclose(old_kernel, new_kernel)
        assert not tf.experimental.numpy.allclose(old_bias, new_bias)

    @abstractmethod
    def check_output_size(self, output):
        pass

    def test_reset_input_spec(self):
        #To Do
        pass

    def test_compute_weighted_parameters(self):
        #Call() calls compute_weighted_parameters()
        layer = self.init_standard_layer()
        output, curve_point_weights= self.call_standard_layer(layer)

        #Alternative calculation of weighted params
        alt_kernel= None
        alt_bias= None

        for i in range(len(layer.curve_kernels)):
            scaled_kernel= curve_point_weights[i] * layer.curve_kernels[i]
            if alt_kernel == None:
                alt_kernel = scaled_kernel
            else:
                alt_kernel += scaled_kernel

        for i in range(len(layer.curve_biases)):
            scaled_bias= curve_point_weights[i] * layer.curve_biases[i]
            if alt_bias == None:
                alt_bias = scaled_bias
            else:
                alt_bias += scaled_bias

        assert tf.experimental.numpy.allclose(layer.kernel, alt_kernel)
        assert tf.experimental.numpy.allclose(layer.bias, alt_bias)

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

    def init_standard_layer(self):
        return Conv2DCurve(self.filters, self.kernel_size, self.fix_points)

    def call_standard_layer(self, conv2d_curve_layer):
        curve_point_weights= self.get_curve_point_weights()
        output= conv2d_curve_layer([tf.random.uniform(shape=self.input_shape), curve_point_weights])
        return output, curve_point_weights

    def test_init(self):    
        conv2d_curve_layer = self.init_standard_layer()
        assert isinstance(conv2d_curve_layer, CurveLayer)
        assert isinstance(conv2d_curve_layer, tf.keras.layers.Conv2D)

    def check_output_size(self, output):
        output_shape_h = self.input_shape[1] - self.kernel_size[0] + 1
        output_shape_w = self.input_shape[2] - self.kernel_size[1] + 1
        assert output.shape == (128, output_shape_h, output_shape_w, 32)


class TestDenseCurveLayer(CurveLayerTest):
    units = 16
    input_shape= (128, 32)  

    def init_standard_layer(self):
        return DenseCurve(self.units, self.fix_points)

    def call_standard_layer(self, dense_curve_layer):
        curve_point_weights= self.get_curve_point_weights()
        output= dense_curve_layer([tf.random.uniform(shape= self.input_shape), curve_point_weights])
        return output, curve_point_weights

    def test_init(self):
        dense_curve_layer = self.init_standard_layer()
        assert isinstance(dense_curve_layer, CurveLayer)
        assert isinstance(dense_curve_layer, tf.keras.layers.Dense)   

    def check_output_size(self, output):
        assert output.shape == (self.input_shape[:-1][0], self.units)


# If we add regularizer, test whether correct kernel is optimized.