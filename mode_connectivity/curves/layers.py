import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CurveLayer(tf.keras.layers.Layer, ABC):
    fix_points: List[bool]
    num_bends: int
    l2: float

    curve_kernels: List[tf.Variable]
    curve_biases: List[tf.Variable]

    def __init__(self, fix_points: List[bool], **kwargs):
        super().__init__(**kwargs)
        self.fix_points = fix_points
        self.num_bends = len(self.fix_points)
        self.l2 = None
        self._reset_input_spec()

    def _reset_input_spec(self):
        """Modify the input specification to take in the curve coefficients as well."""
        self.input_spec = [
            self.input_spec,
            tf.keras.layers.InputSpec(shape=((self.num_bends,))),
        ]

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
        self.curve_kernels = [None] * len(self.fix_points)
        self.curve_biases = [None] * len(self.fix_points)
        for i, fixed in enumerate(self.fix_points):
            self.curve_kernels[i] = self._add_kernel(i, kernel_shape, fixed)
            if self.use_bias:
                self.curve_biases[i] = self._add_bias(i, bias_shape, fixed)

    def _add_kernel(self, index: int, shape: Tuple, fixed: bool):
        return self._add_weights(index, shape, fixed, "kernel")

    def _add_bias(self, index: int, shape: Tuple, fixed: bool):
        return self._add_weights(index, shape, fixed, "bias")

    def _add_weights(self, index: int, shape: Tuple, fixed: bool, parameter_type: str):
        name = f"{parameter_type}_curve_{index}"
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

    def compute_weighted_parameters(
        self, curve_point_weights: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute weights for the curve kernel and bias.

        Args:
            curve_point_weights (tf.Tensor): Coefficients calculated from the curve.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The scaled weights for kernel and bias.
        """
        if self.l2 is None:
            self.l2 = tf.Variable(0.0, trainable=False)
        weights = (
            self._compute_single_parameter(self.curve_kernels, curve_point_weights),
            self._compute_single_parameter(self.curve_biases, curve_point_weights),
        )
        return weights

    def _compute_single_parameter(
        self, parameters: List[tf.Variable], curve_point_weights: tf.Tensor
    ) -> tf.Tensor:
        """Multiplies the given weights by the respective coefficient
        and adds them together.

        Adds to the l2 loss as well.

        Args:
            weights (Dict[int, tf.Variable]): The weights.
            curve_point_weights (tf.Tensor): Coefficients calculated from the curve.

        Returns:
            tf.Tensor: The multiplied weights.
        """
        combined_params = tf.stack([w.value() for w in parameters], axis=-1)
        params_averaged = tf.linalg.matvec(combined_params, curve_point_weights)

        # weight_sum = 0
        # for i in range(curve_point_weights.shape[0]):
        #     weight_sum += weights[i].value() * curve_point_weights[i]

        if params_averaged is not None:
            # I think this should always be called
            self.l2.assign_add(tf.reduce_sum(params_averaged**2))
        return params_averaged


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
        tf.keras.layers.Conv2D.build(self, input_shape[0])
        self._reset_input_spec()
        # Register curve kernels and biases
        self.add_parameter_weights(
            kernel_shape=self.kernel.shape, bias_shape=self.bias.shape
        )

    # TODO Check inputs as Tuple or seperate
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        x, curve_point_weights = inputs
        self.kernel, self.bias = self.compute_weighted_parameters(curve_point_weights)
        return tf.keras.layers.Conv2D.call(self, x)


class DenseCurve(CurveLayer, tf.keras.layers.Dense):
    def __init__(
        self,
        units,
        fix_points: List[bool],
        **kwargs,
    ):
        super().__init__(units=units, fix_points=fix_points, **kwargs)

    def build(self, input_shape):
        tf.keras.layers.Dense.build(self, input_shape[0])
        self._reset_input_spec()
        # Register curve kernels and biases
        self.add_parameter_weights(
            kernel_shape=self.kernel.shape, bias_shape=self.bias.shape
        )

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        x, curve_point_weights = inputs
        self.kernel, self.bias = self.compute_weighted_parameters(curve_point_weights)
        return tf.keras.layers.Dense.call(self, x)
