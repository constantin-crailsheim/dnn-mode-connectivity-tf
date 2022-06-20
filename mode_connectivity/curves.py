import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np
from scipy.special import binom

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




class Dense(CurveLayer, tf.keras.layers.Dense):
    def __init__(self,
                 units,
                 activation = None,
                 use_bias = True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 activity_regularizer = None, 
                 kernel_constraint = None,
                 bias_constaint = None,
                 fix_points: List[bool],
                 **kwargs, 
                 
    ):
        super(Dense, self).__init__(
            units = units,
            fix_points = fix_points,
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer,
            **kwargs
        )

        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(f'Received an invalid value for `units`, expected '
                       f'a positive integer. Received: units={units}')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        tf.keras.layers.Dense.build(self, input_shape)
        input_shape = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        kernel_shape = self.kernel_size + (input_channel // self.groups, self.filters)
        bias_shape = (self.filters,)
        last_dim = tf.compat.dimension_value(input_shape[-1])
       
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to a Dense layer '
                             'should be defined. Found None. '
                             f'Full input shape received: {input_shape}')
       
        self.add_parameter_weights(kernel_shape = kernel_shape, bias_shape = bias_shape)


    def call(self, inputs, coeffs_t: tf.Tensor):
        self.kernel, self.bias = self.compute_weights_t(coeffs_t)   
        return tf.matmul(inputs, self.kernel) + self.bias

                  



#class CurveNet(tf.keras.Model):
#    def __init__(self, num_classes, curve, architecture, num_bends, 
#                 fix_start = True, fix_end = True, architecture_kwargs={}):
#        super().__init__() 
#        self.num_classes = num_classes
#        self.num_bends = num_bends
#        self.fix_points = [fix_start] + [False]*(self.num_bends - 2) + [fix_end]
#        
#        self.curve = curve
#        self.architecture = architecture#
#
#        self.l2 = 0.0
#        self.coeff_layer = self.curve(self.num_bends)
#        self.net = self.architecture(num_classes, fix_points = self.fix_points, **architecture_kwargs)
#        self.curve_modules = []
#        for module in self.net.modules():