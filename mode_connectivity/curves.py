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

class Conv2d(CurveModule):
    def __init__(self,
                fix_points,
                filters,
                kernel_size,
                strides=(1, 1),
                padding='valid',
                data_format=None,
                dilation_rate=(1, 1),
                groups=1,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                **kwargs):
                #Should there be a possibility to determine input dimensionality as well?
        super().__init__(fix_points, ('kernel', 'bias'))

        # if filters % groups != 0:
        #     raise ValueError('Filters must be divisible by groups')

        # Sources:   1) https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/layers/convolutional/base_conv.py#L28
        #            2) https://github.com/keras-team/keras/blob/v2.9.0/keras/layers/convolutional/conv2d.py#L28-L188

        #Copied from 1)
        self.rank = 2
        if isinstance(filters, float):
            filters = int(filters)
        if filters is not None and filters <= 0:
            raise ValueError('Invalid value for argument `filters`. '
                        'Expected a strictly positive value. '
                        f'Received filters={filters}.')
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(
            strides, self.rank, 'strides', allow_zero=True)
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, self.rank, 'dilation_rate')
        self.groups = groups or 1
        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self._tf_data_format = conv_utils.convert_data_format(self.data_format, self.rank + 2)

        self._validate_init()

    def build(self, input_shape):
        #Copied from 1) build(self, input_shape)
        #Built gets called once when call() is called for the first time.
        input_shape = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by the number '
                'of groups. Received groups={}, but the input has {} channels '
                '(full input shape is {}).'.format(self.groups, input_channel,
                                                    input_shape))
        kernel_shape = self.kernel_size + (input_channel // self.groups, self.filters)

        self.compute_output_shape(input_shape)

        #Substitutes register_parameter and reset_parameters steps 
        for i, fixed in enumerate(self.fix_points):
            temp_kernel_name = "kernel_" + str(i) 
            temp_kernel_obj = self.add_weight(
                name='kernel',
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable= not fixed,
                dtype=self.dtype)
            setattr(self, temp_kernel_name, temp_kernel_obj)

            temp_bias_name = "bias_" + str(i)
            if self.use_bias:
                temp_bias_obj = self.add_weight(
                    name='bias',
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    trainable= not fixed,
                    dtype=self.dtype)
                setattr(self, temp_bias_name, temp_bias_obj)
            else:
                setattr(self, temp_bias_name, None)

        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})
        self.built = True

    def call(self, inputs, coeffs_t: tf.Tensor):
        #Source 1) convolution_op(self, inputs, kernel)
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, str):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding

        kernel_t, bias_t= self.compute_weights_t(coeffs_t)

        outputs = tf.nn.convolution(
            inputs,
            kernel_t,
            strides=list(self.strides),
            padding=tf_padding,
            dilations=list(self.dilation_rate),
            data_format=self._tf_data_format,
            name=self.__class__.__name__)

        if self.use_bias:
            output_rank = outputs.shape.rank
            if output_rank is not None and output_rank > 2 + self.rank:
                def _apply_fn(o):
                    return tf.nn.bias_add(o, bias_t, data_format=self._tf_data_format)
                outputs = conv_utils.squeeze_batch_dims(outputs, _apply_fn, inner_rank=self.rank + 1)
            else:
                outputs = tf.nn.bias_add(outputs, bias_t, data_format=self._tf_data_format)

        if self.activation is not None:
            outputs= self.activation(outputs)
        return outputs

    def _validate_init(self):
        #Copied from 1) _validate_init(self) but pruned
        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
            'The number of filters must be evenly divisible by the number of '
            'groups. Received: groups={}, filters={}'.format(
                self.groups, self.filters))

        if not all(self.kernel_size):
            raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                        'Received: %s' % (self.kernel_size,))

        if not all(self.strides):
            raise ValueError('The argument `strides` cannot contains 0(s). '
                        'Received: %s' % (self.strides,))

    #Preferably import script in source 1) and call method from import
    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. '
                       f'The input_shape received is {input_shape}, '
                       f'where axis {channel_axis} (0-based) '
                       'is the channel dimension, which found to be `None`.')
        return int(input_shape[channel_axis])

    #Preferably import script in source 1) and call method from import
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        try:
            if self.data_format == 'channels_last':
                return tf.TensorShape(
                    input_shape[:batch_rank] +
                    self._spatial_output_shape(input_shape[batch_rank:-1]) +
                    [self.filters])
            else:
                return tf.TensorShape(
                    input_shape[:batch_rank] + [self.filters] +
                    self._spatial_output_shape(input_shape[batch_rank + 1:]))

        except ValueError:
            raise ValueError(
                f'One of the dimensions in the output is <= 0 '
                f'due to downsampling in {self.name}. Consider '
                f'increasing the input size. '
                f'Received input shape {input_shape} which would produce '
                f'output shape with a zero or negative value in a '
                f'dimension.')

    #Preferably import script in source 1) and call method from import
    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return -1 - self.rank
        else:
            return -1

    #Preferably import script in source 1) and call method from import
    def _spatial_output_shape(self, spatial_input_shape):
        return [
            conv_utils.conv_output_length(  # pylint: disable=g-complex-comprehension
                length,
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            for i, length in enumerate(spatial_input_shape)
            ]