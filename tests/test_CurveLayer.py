import tensorflow as tf
import numpy as np

from mode_connectivity.curves.layers import CurveLayer, Conv2DCurve, DenseCurve
from mode_connectivity.curves.curves import Bezier

from typing import List
from abc import ABC, abstractmethod
import pytest


class TestCurveLayer:
    def __init__(self):
        self.fix_points = [True, False, True]


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
        #Tests the __init__ method and inheritance
        pass

    def test_init_curve(self):
        curve_layer = CurveLayer([True, False, True])
        assert isinstance(curve_layer, tf.keras.layers.Layer)


    def test_build(self):
        layer = self.init_standard_layer()
        #Also calls build()
        self.call_standard_layer(layer)
        
        self.test_built_layer(layer) 

    def test_built_layer(self, built_layer):
        # Build in parent class tf.keras.layers.Conv2 should add kernel and bias parameters 
        # (Even though they won't be used directly)        
        assert built_layer.kernel != None
        assert built_layer.bias != None

        # Conv2DCurve build() should add curve_kernels and curve_biases parameters    
        assert built_layer.curve_kernels != None
        assert built_layer.curve_biases != None

        for (param, curve_param) in [("kernel" ,"curve_kernels"), ("bias", "curve_biases")]:
            assert type(getattr(built_layer, curve_param)) == List
            assert len(getattr(built_layer, curve_param)) == len(self.fix_points)
            for i in range(self.fix_points): #Fix points
                assert getattr(built_layer, curve_param)[i].shape == getattr(built_layer, param).shape
                if param != "bias":
                    assert tf.math.count_nonzero(getattr(built_layer, curve_param)[i]) != 0 #Check if initialized


    def test_call(self):
        layer = self.init_standard_layer()
        self.test_param_updates(layer)

        # Even without optimization of the network parameters, the params should be updated
        # since we change the location on the curve/ randomly set curve_point_weights of the submodels.
        self.test_param_updates(layer)

        output, curve_point_weights= self.call_standard_layer(layer)
        self.check_output_size(layer, output)

    def test_param_updates(self, layer):
        old_kernel= layer.kernel
        old_bias= layer.bias
        self.call_standard_layer(layer)
        new_kernel= layer.kernel
        new_bias= layer.bias

        assert not np.close(old_kernel, new_kernel)
        assert not np.close(old_bias, new_bias)

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
        for i in range(len(layer.curve_kernels)):
            scaled_kernel= curve_point_weights[i] * layer.curve_kernels[i]
            if alt_kernel == None:
                alt_kernel = scaled_kernel
            else:
                alt_kernel += scaled_kernel

        for i in range(len(layer.curve_biases)):
            scaled_bias= curve_point_weights[i] * layer.curve_kernels[i]
            if alt_bias == None:
                alt_bias = scaled_bias
            else:
                alt_bias += scaled_bias

        assert np.allclose(layer.kernel, alt_kernel)
        assert np.allclose(layer.bias, alt_bias)


    # def test_compute_weighted_parameters(self):
    #     shape = (3, 3, 1, 32)
    #     num_bends = 3
    #     weights = [tf.random.uniform(shape) for _ in range(num_bends)]
    #     curve_point_weights = tf.random.uniform([num_bends])

    #     # New method (matrix multiplication)
    #     combined_weights = tf.stack([w for w in weights], axis=-1)
    #     weights_avg_new = tf.linalg.matvec(combined_weights, curve_point_weights)

    #     assert weights_avg_new.shape == shape

    #     # Old method (Looping)
    #     weights_avg_old = 0
    #     for i in range(curve_point_weights.shape[0]):
    #         weights_avg_old += weights[i] * curve_point_weights[i]

    #     assert weights_avg_old.shape == shape

    #     # Assert both give equal results
    #     assert np.allclose(weights_avg_new, weights_avg_old)


    # Still necessary if we have test_init?
    # def test_init_direct(self): 
    #     with pytest.raises(
    #         TypeError,
    #         match="Can't instantiate abstract class CurveLayer with abstract methods build, call",
    #     ):
    #         CurveLayer([True, False, True])


class TestConv2DCurveLayer(TestCurveLayer):
    def init_standard_layer(self):
        filters=32
        kernel_size=(3, 3)
        return Conv2DCurve(filters, kernel_size, self.fix_points)

    def call_standard_layer(self, conv2d_curve_layer):
        curve_point_weights= self.get_curve_point_weights()
        output= conv2d_curve_layer([tf.random.uniform(shape=(128, 28, 28, 1)), curve_point_weights])
        return output, curve_point_weights

    def test_init(self):    
        conv2d_curve_layer = self.init_standard_layer()
        assert isinstance(conv2d_curve_layer, CurveLayer)
        assert isinstance(conv2d_curve_layer, tf.keras.layers.Conv2D)

    def check_output_size(self, output):
        conv_output_size = 28-3+1 #Image size - Kernel size + 1
        assert output.shape == (128, 32, conv_output_size, conv_output_size, 1)

    #Wird hier test_build und test_call auch für TestConv2DCurveLayer und TestDenseCurveLayer aufgerufen?


class TestDenseCurveLayer(TestCurveLayer):
    def init_standard_layer(self):
        units=16
        return DenseCurve(units, self.fix_points)

    def call_standard_layer(self, dense_curve_layer):
        curve_point_weights= self.get_curve_point_weights()
        output= dense_curve_layer([tf.random.uniform(shape=(128, 32)), curve_point_weights])
        return output, curve_point_weights

    def test_init(self):
        dense_curve_layer = self.init_standard_layer()
        assert isinstance(dense_curve_layer, CurveLayer)
        assert isinstance(dense_curve_layer, tf.keras.layers.Dense)   

    def check_output_size(self, output):
        assert output.shape == (128, 16)


# If we add regularizer, test whether correct kernel is optimized.