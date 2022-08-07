from abc import ABC
from typing import List, Tuple, Type, Union

import tensorflow as tf


class CurveLayer(tf.keras.layers.Layer, ABC):
    """
    Base class for the Curve implementations of TensorFlow-Layers.
    A Curve consists of several bends that are represented by a list of parameters (kernels and biases).
    For each of those bends it can be specified if it is trainable or not/ fixed.
    """
    fix_points: List[bool]
    num_bends: int

    curve_kernels: List[tf.Variable]
    curve_biases: List[tf.Variable]

    def __init__(
        self, fix_points: List[bool], base_layer: Type[tf.keras.layers.Layer], **kwargs
    ):
        """
        Initializes the CurveLayer.

        Args:
            fix_points (List[bool]): List indicating for each bend on the curve if it is not trainable.
            base_layer (Type[tf.keras.layers.Layer]): Regular TensorFlow-Layer.

        Raises:
            ValueError: Indicates if the amount of bends is misspecified.
        """
        if len(fix_points) < 2:
            raise ValueError(
                f"You need to specify at least two bends (found {len(fix_points)})!"
            )

        super().__init__(**kwargs)
        self.base_layer = base_layer
        self.fix_points = fix_points
        self.num_bends = len(self.fix_points) - 2
        self._reset_input_spec()

    def _reset_input_spec(self):
        """Modify the input shape specification to account for the bends."""
        self.input_spec = [
            self.input_spec,
            tf.keras.layers.InputSpec(shape=((len(self.fix_points),))),
        ]

    def build(self, input_shape):
        """
        Invoked by the first __call__() to the layer.
        Determines the shape of the parameters of the layer (kernel and bias) and registers them.

        Args:
            input_shape
        """

        # We need to remove the kernel and bias regularizer temporarly for building
        # or otherwise the kernel/bias-regularizer will get registered and thus
        # taken into account when calculating the loss.
        kernel_reg, bias_reg = self.kernel_regularizer, self.bias_regularizer
        self.kernel_regularizer, self.bias_regularizer = None, None
        self.base_layer.build(self, input_shape[0])
        self.kernel_regularizer, self.bias_regularizer = kernel_reg, bias_reg
        self._reset_input_spec()
        # Register curve kernels and biases
        self.add_parameter_weights(
            kernel_shape=self.kernel.shape, bias_shape=self.bias.shape
        )
        del self.kernel
        del self.bias

    # TODO Check inputs as Tuple or seperate
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        """
        Applies the CurveLayer to inputs.
        As parameters it uses a weighted sum of the parameters of the CurveLayer/ bends.
        This corresponds to the model lying on the respective point on the curve.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): Tuple of layer inputs and weights of the bends.

        Returns:
            _type_: Layer output
        """
        x, curve_point_weights = inputs
        self.kernel, self.bias = self.compute_weighted_parameters(curve_point_weights)
        return self.base_layer.call(self, x)

    def add_parameter_weights(self, kernel_shape: Tuple, bias_shape: Tuple):
        """
        Adds parameters (kernel and bias) for each bend.
        Called in the build() method of the  CurveLayer.

        The kernel and bias variables are saved in dictionaries as
        self.curve_kernels and self.curve_biases, with
        the index of the bend as the respective key.

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
        """
        Adds the kernel for a certain bend.

        Args:
            index (int): Index of the bend.
            shape (Tuple): Shape of the kernel.
            fixed (bool): Boolean indicating if the kernel is not trainable.

        Returns:
            _type_: Resulting kernel
        """
        name = f"kernel_curve_{index}"
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

    def _add_bias(self, index: int, shape: Tuple, fixed: bool):
        """
        Adds the bias for a certain bend.

        Args:
            index (int): Index of the bend.
            shape (Tuple): Shape of the bias.
            fixed (bool): Boolean indicating if the bias is not trainable.

        Returns:
            _type_: Resulting bias
        """

        name = f"bias_curve_{index}"
        weight = self.add_weight(
            name=name,
            shape=shape,
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=not fixed,
            dtype=self.dtype,
        )
        return weight

    def compute_weighted_parameters(
        self, curve_point_weights: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute reweighted parameters (kernel and bias) based on the weights for each curve point/ bend.

        Args:
            curve_point_weights (tf.Tensor): Coefficients for each curve point/ bend calculated from the curve.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Reweighted kernel and bias.
        """
        weights = (
            self._compute_single_parameter(self.curve_kernels, curve_point_weights),
            self._compute_single_parameter(self.curve_biases, curve_point_weights),
        )
        return weights

    def _compute_single_parameter(
        self, parameters: List[tf.Variable], curve_point_weights: tf.Tensor
    ) -> tf.Tensor:
        """
        Multiplies the parameters of the curve points/ bends by the respective weights and adds them up.

        Args:
            parameters (List[tf.Variable]): Parameters (kernels and biases) of the curve points/ bends.
            curve_point_weights (tf.Tensor): Weights of the curve points/ bends.

        Returns:
            tf.Tensor: Reweighted parameter.
        """

        combined_params = tf.stack([w.value() for w in parameters], axis=-1)
        params_averaged = tf.linalg.matvec(combined_params, curve_point_weights)
        return params_averaged


class Conv2DCurve(CurveLayer, tf.keras.layers.Conv2D):
    """Implementation of the Conv2D-Layer as CurveLayer."""
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
            base_layer=tf.keras.layers.Conv2D,
            **kwargs,
        )


class DenseCurve(CurveLayer, tf.keras.layers.Dense):
    """Implementation of the Dense-Layer as CurveLayer."""
    def __init__(
        self,
        units,
        fix_points: List[bool],
        **kwargs,
    ):
        super().__init__(
            units=units,
            fix_points=fix_points,
            base_layer=tf.keras.layers.Dense,
            **kwargs,
        )