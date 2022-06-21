from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
import tensorflow as tf
from scipy.special import binom

from .utils import split_list


class Curve(tf.keras.layers.Layer, ABC):
    num_bends: int

    def __init__(self, num_bends: int):
        super().__init__()
        self.num_bends = num_bends

    @abstractmethod
    def call(self, uniform_tensor: tf.Tensor):
        pass


class Bezier(Curve):
    def __init__(self, num_bends: int):
        super().__init__(num_bends=num_bends)
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


class CurveNet(tf.keras.Model):
    num_classes: int
    num_bends: int
    l2: float

    fix_points: List[bool]

    curve: Curve
    curve_model: tf.keras.Model

    def __init__(
        self,
        num_classes: int,
        num_bends: int,
        curve: Type[Curve],  # Bezier, Polychain
        curve_model: Type[tf.keras.Model],  # ConvFCCurve, VGGCurve
        fix_start: bool = True,
        fix_end: bool = True,
        architecture_kwargs: Union[Dict[str, Any], None] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_bends = num_bends
        self.fix_points = [fix_start] + [False] * (self.num_bends - 2) + [fix_end]
        self.l2 = 0.0

        self.curve = curve(self.num_bends)
        self.curve_model = curve_model(
            num_classes=num_classes, fix_points=self.fix_points, **architecture_kwargs
        )
        self.curve_layers = [
            layer
            for layer in self.curve_model.submodules()  # TODO Need to check if this gathers the correct modules
            if issubclass(layer.__class__, CurveLayer)
        ]

    def import_base_parameters(self, base_model: tf.keras.Model, index: int) -> None:
        """Import parameters from the base model into this model.

        Args:
            base_model (tf.keras.Model): The base model from which parameters should be imported.
            index (int): _description_
        """
        variables: List[tf.Variable] = self.curve_model.variables[
            index :: self.num_bends
        ]
        base_variables: List[tf.Variable] = base_model.variables

        # or should we call get/set_weights ?
        # would be even better to initialize by names
        # the indexing [index :: self.num_bends] above is not so clear
        for variable, base_variable in zip(variables, base_variables):
            variable.assign(base_variable.value)

    def init_linear(self) -> None:
        """Initialize the linear layer of the model."""
        split_weights = split_list(self.curve_model.variables, size=self.num_bends)
        for weights in split_weights:
            self._compute_inner_weights(weights=weights)

    def _compute_inner_weights(self, weights: List[tf.Variable]) -> None:
        # Is this procedure mentioned somewhere in the paper?
        first_weight, last_weight = weights[0].value, weights[-1].value
        for i in range(1, self.num_bends - 1):
            alpha = i * 1.0 / (self.num_bends - 1)
            weights[i].assign(alpha * first_weight + (1.0 - alpha) * last_weight)

    def _compute_l2(self) -> None:
        """Compute L2 for each of the curve modules and sum up."""
        self.l2 = sum(module.l2 for module in self.curve_layers)

    def call(
        self,
        inputs: tf.Tensor,
        uniform_tensor: Union[tf.Tensor, None] = None,
        training=None,
        mask=None,
    ):
        # Renamed 't' to 'uniform_tensor' for clarity
        # TODO find a better name for uniform_tensor and coeffs_t
        if uniform_tensor is None:
            uniform_tensor = tf.random.uniform(shape=(1,), dtype=inputs.dtype)
        coeffs_t = self.curve(uniform_tensor)
        output = self.curve_model(inputs, coeffs_t)
        self._compute_l2()
        return output

    def import_base_buffers(self, base_model: tf.keras.Model) -> None:
        # Not needed for now, only used in test_curve.py
        raise NotImplementedError()

    def export_base_parameters(self, base_model: tf.keras.Model, index: int) -> None:
        # Not needed for now, actually never used in original repo
        raise NotImplementedError()

    def weights(self, inputs: tf.Tensor):
        # Not needed for now, only called in eval_curve.py and plane.py
        raise NotImplementedError()


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


class DenseCurve(CurveLayer, tf.keras.layers.Dense):
    def __init__(
        self,
        units,
        fix_points: List[bool],
        **kwargs,
    ):
        super().__init__(units=units, fix_points=fix_points, **kwargs)

    def build(self, input_shape):
        tf.keras.layers.Dense.build(self, input_shape)
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        kernel_shape = [last_dim, self.units]
        bias_shape = [
            self.units,
        ]

        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )

        self.add_parameter_weights(kernel_shape=kernel_shape, bias_shape=bias_shape)

    def call(self, inputs, coeffs_t: tf.Tensor):
        self.kernel, self.bias = self.compute_weights_t(coeffs_t)
        return tf.keras.layers.Dense(self, inputs)
