from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
from scipy.special import binom


class Bezier(tf.keras.Model):
    def __init__(self, num_bends: int):
        super().__init__()
        self.binom = tf.Variable(
            tf.constant(binom(num_bends - 1, np.arange(num_bends), dtype=np.float32)),
            trainable=False,
        )
        self.range = tf.Variable(tf.range(0, float(num_bends)), trainable=False)
        self.rev_range = tf.Variable(
            tf.range(float(num_bends - 1), -1, delta=-1), trainable=False
        )

        # Not sure if this is the best way to substitute register_buffer() in PyTorch
        # The PyTorch Buffer in this example is not considered a model parameter, not trained,
        # part of the module's state, moved to cuda() or cpu() with the rest of the model's parameters

    def call(self, t: float):
        return (
            self.binom
            * tf.math.pow(t, self.range)
            * tf.math.pow((1.0 - t), self.rev_range)
        )


class CurveLayer(tf.keras.layers.Layer, ABC):
    fix_points: List[bool]
    num_bends: int
    l2: float

    def __init__(self, fix_points: List[bool], **kwargs):
        super().__init__(**kwargs)
        self.fix_points = fix_points
        self.num_bends = len(self.fix_points)
        self.l2 = 0.0

    @abstractmethod
    def build(self, *args, **kwargs):
        pass

    @abstractmethod
    def call(self, *args, **kwargs):
        pass

    def add_parameter_weights(self, kernel_shape: Tuple, bias_shape: Tuple):
        """Add kernel and bias weights for each curve point.

        This method needs to be called in the build() method of
        the new layer.

        The kernel and bias variables are saved in dictionaries in
        self.curve_kernels and self.curve_biases, with
        the index of the curve point as the respective key.

        Args:
            kernel_shape (Tuple): Shape of the kernel.
            bias_shape (Tuple): Shape of the bias.
        """
        self.curve_kernels = {}
        self.curve_biases = {}
        for i, fixed in enumerate(self.fix_points):
            self.curve_kernels[i] = self._add_kernel(i, kernel_shape, fixed)
            if self.use_bias:
                self.curve_biases[i] = self._add_bias(i, bias_shape, fixed)

    def _add_kernel(self, index: int, shape: Tuple, fixed: bool):
        return self._add_weights(index, shape, fixed, "kernel")

    def _add_bias(self, index: int, shape: Tuple, fixed: bool):
        return self._add_weights(index, shape, fixed, "bias")

    def _add_weights(self, index: int, shape: Tuple, fixed: bool, parameter_type: str):
        name = f"curve_{parameter_type}_{index}"
        weight = self.add_weight(
            name=name,
            shape=shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=not fixed,
            dtype=self.dtype,
        )
        return weight

    def compute_weights_t(self, coeffs_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute weights for the curve kernel and bias.

        Args:
            coeffs_t (tf.Tensor): Coefficients calculated from the curve.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The scaled weights for kernel and bias.
        """
        self.l2 = 0.0
        weights = (
            self._compute_single_weights(self.curve_kernels, coeffs_t),
            self._compute_single_weights(self.curve_biases, coeffs_t),
        )
        return weights

    def _compute_single_weights(
        self, weights: Dict[int, tf.Variable], coeffs_t: tf.Tensor
    ) -> tf.Tensor:
        """Multiplies the given weights by the respective coefficient
        and adds them together.

        Adds to the l2 loss as well.

        Args:
            weights (Dict[int, tf.Variable]): The weights.
            coeffs_t (tf.Tensor): Coefficients calculated from the curve.

        Raises:
            ValueError: If the length of weights and coefficients does not add up.

        Returns:
            tf.Tensor: The multiplied weights.
        """
        if len(weights) != len(coeffs_t):
            raise ValueError(
                f"Lengths of curve weights {len(weights)} and coefficients {len(coeffs_t)} is not equal!"
                + f"Parameters: {[w.name for w in weights.values()]}"
            )

        # I think this could also be solved by matrix multiplication.
        weight_sum = 0
        for i, coeff in enumerate(coeffs_t):
            weight_sum += weights[i].value() * coeff

        if weight_sum is not None:
            # I think this should always be called
            self.l2 += tf.reduce_sum(weight_sum**2)
        return weight_sum


class Conv2DCurve(CurveLayer, tf.keras.layers.Conv2D):
    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        fix_points: List[bool],
        **kwargs,
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            fix_points=fix_points,
            **kwargs,
        )

    def build(self, input_shape):
        tf.keras.layers.Conv2D.build(self, input_shape)
        # Built gets called once when call() is called for the first time.
        input_shape = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        kernel_shape = self.kernel_size + (input_channel // self.groups, self.filters)
        bias_shape = (self.filters,)

        # Register curve kernels and biases
        self.add_parameter_weights(kernel_shape=kernel_shape, bias_shape=bias_shape)

    def call(self, inputs, coeffs_t: tf.Tensor):
        self.kernel, self.bias = self.compute_weights_t(coeffs_t)
        return tf.keras.layers.Conv2D.call(self, inputs)
