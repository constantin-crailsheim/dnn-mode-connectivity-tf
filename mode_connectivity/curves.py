import tensorflow as tf
import numpy as np
from scipy.special import binom
from typing import List

from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import conv_utils

class Bezier(tf.keras.Model):
    def __init__(self, num_bends: int):
        super().__init__()
        self.binom = tf.Variable(tf.constant(binom(num_bends - 1, np.arange(num_bends), dtype=np.float32)), trainable= False)
        self.range = tf.Variable(tf.range(0, float(num_bends)), trainable= False)
        self.rev_range = tf.Variable(tf.range(float(num_bends - 1), -1, delta= -1), trainable= False)

        # Not sure if this is the best way to substitute register_buffer() in PyTorch
        # The PyTorch Buffer in this example is not considered a model parameter, not trained, 
        # part of the module's state, moved to cuda() or cpu() with the rest of the model's parameters

    def call(self, t: float):
        return self.binom * \
            tf.math.pow(t, self.range) * \
            tf.math.pow((1.0 - t), self.rev_range)


class CurveModule(tf.keras.Model):
    def __init__(self, fix_points: List[bool], parameter_types=('weight', 'bias')):
        super().__init__()
        self.fix_points = fix_points
        self.num_bends = len(self.fix_points)
        self.parameter_types = parameter_types #Changed variable name from parameter_names to parameter_types. The former could be confusing.
        self.l2 = 0.0

    def compute_weights_t(self, coeffs_t: tf.Tensor):
        w_t = [None] * len(self.parameter_types) #e.g [None, None] for Weight and Bias
        self.l2 = 0.0
        for i, parameter_type in enumerate(self.parameter_types): #e.g iterates [(0, Weight), (1, Bias)]
            for j, coeff in enumerate(coeffs_t): #e.g [(0, 0.3), (1, 0.4), (2, 0.3)] with coeffs as the weights of the respective sub-models
                parameter = getattr(self, '%s_%d' % (parameter_type, j)) #Get Weight or Bias tensor of respective sub_model
                if parameter is not None:
                    if w_t[i] is None:
                        w_t[i] = parameter * coeff
                    else:
                        w_t[i] += parameter * coeff
            if w_t[i] is not None:
                self.l2 += tf.reduce_sum(w_t[i] ** 2)
        return w_t