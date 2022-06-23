import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
import tensorflow as tf
from scipy.special import binom

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Why do we
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
    curve_layers: List[tf.keras.layers.Layer]  # CurveLayer

    def __init__(
        self,
        num_classes: int,
        num_bends: int,
        weight_decay: float,
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
        # Since we use l2 in computation late, we need to instantiate it as a tf.Variable
        # https://www.tensorflow.org/guide/function#creating_tfvariables
        self.l2 = None

        self.curve = curve(self.num_bends)
        self.curve_model = curve_model(
            num_classes=num_classes,
            fix_points=self.fix_points,
            weight_decay=weight_decay,
            **architecture_kwargs,
        )
        self.curve_layers = [
            layer
            for layer in self.curve_model.submodules
            if issubclass(layer.__class__, CurveLayer)
        ]

    def import_base_parameters(self, base_model: tf.keras.Model, index: int) -> None:
        """Import parameters from the base model into this model.

        Parameters in the Base Model (without Curve layers) are
        saved with the name
            <layerName>_<layerIndex>/<parameter>:0

        For example:
            conv2d/kernel:0
            conv2d/bias:0
            dense_1/kernel:0

        Whereas parameters in the Curve Model are save with a '_curve' suffix
        and an index that matches them to the curve point.

        For example:
            conv2d_curve/kernel_curve_1:0
            conv2d_curve/bias_curve_0:0
            dense_1_curve/kernel_curve_2:0

        Here we want to load weights for a specific point on the curve
        from the pretrained Base Model.

        Args:
            base_model (tf.keras.Model): The base model from which parameters should be imported.
            index (int): Index of the curve point.
        """
        if not self.curve_model.built:
            self._build_from_base_model(base_model)

        weights = {w.name: w for w in self.curve_model.variables}
        base_weights = {w.name: w for w in base_model.variables}

        assigned_weights = []
        for name, weight in weights.items():
            parameter_index = self._find_parameter_index(name)
            if parameter_index != index:
                # Kernel/Bias index doesn't match the curve index.
                # What is the implication?
                logger.debug(
                    f"Index of {name} does not match kernel/bias index."
                )
                continue

            base_name = self._get_base_name(name, index)
            base_weight = base_weights.get(base_name)
            if base_weight is None:
                logger.debug(
                    f"Could not assign to weight {name} (base_name: {base_name})"
                )
                continue

            weight.assign(base_weight.value())
            assigned_weights.append(f"{base_name} -> {name}")

        logger.info(
            f"Assigned weights for point #{index}: {', '.join(assigned_weights)}"
        )

    def _build_from_base_model(self, base_model: tf.keras.Model):
        """Build the model to initialize weights."""
        base_input_shape = base_model.layers[0].input_shape
        coeffs_t_input_shape = (self.num_bends,)
        input_shape = [
            tf.TensorShape(base_input_shape),
            tf.TensorShape(coeffs_t_input_shape),
        ]
        self.curve_model.build(input_shape)

    @staticmethod
    def _find_parameter_index(parameter_name: str) -> Union[int, None]:
        """Finds the index (respective curve point) of a curve parameter name."""
        results = re.findall(r"(\d+):", parameter_name)
        if not results:
            return None
        return int(results[0])

    @staticmethod
    def _get_base_name(parameter_name: str, index: int):
        """Returns the base model parameter name, given a curve model parameter name."""
        return parameter_name.replace("_curve", "").replace(f"_{index}:", ":")

    def init_linear(self) -> None:
        """Initialize the linear inner curve weights of the model."""
        # TODO What does 'init_linear' mean? This does not initialize a linear layer.
        # Initialize linear means the the inner points of the curve are initialized as
        # linearly between the end points, depending on how many bends we have.
        for layer in self.curve_layers:
            self._compute_inner_weights(weights=layer.curve_kernels)
            self._compute_inner_weights(weights=layer.curve_biases)

    def _compute_inner_weights(self, weights: List[tf.Variable]) -> None:
        # Is this procedure mentioned somewhere in the paper?
        first_weight, last_weight = weights[0].value(), weights[-1].value()
        for i in range(1, self.num_bends - 1):
            alpha = i * 1.0 / (self.num_bends - 1)
            weights[i].assign(alpha * first_weight + (1.0 - alpha) * last_weight)

    def _compute_l2(self) -> None:
        """Compute L2 for each of the curve modules and sum up."""
        if self.l2 is None:
            self.l2 = tf.Variable(0.0, trainable=False)
        self.l2.assign(sum(module.l2 for module in self.curve_layers))

    def call(
        self,
        inputs: tf.Tensor,  # Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]],
        uniform_tensor: Union[tf.Tensor, None] = None,
        training=None,
        mask=None,
    ):
        # Renamed 't' to 'uniform_tensor' for clarity
        # TODO find a better name for uniform_tensor and coeffs_t
        # if isinstance(inputs, tuple):
        #     inputs, uniform_tensor = inputs
        # else:
        #     uniform_tensor = tf.random.uniform(shape=(1,), dtype=inputs.dtype)
        if uniform_tensor is None:
            uniform_tensor = tf.random.uniform(shape=(1,), dtype=inputs.dtype)
        coeffs_t = self.curve(uniform_tensor)
        output = self.curve_model((inputs, coeffs_t))
        self._compute_l2()
        return output

    def import_base_buffers(self, base_model: tf.keras.Model) -> None:
        # Not needed for now, only used in test_curve.py
        raise NotImplementedError()

    def export_base_parameters(self, base_model: tf.keras.Model, index: int) -> None:
        # Not needed for now, actually never used in original repo
        raise NotImplementedError()

    # def weights(self, inputs: tf.Tensor):
    #     # Not needed for now, only called in eval_curve.py and plane.py
    #     raise NotImplementedError()


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

    def compute_weights_t(self, coeffs_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute weights for the curve kernel and bias.

        Args:
            coeffs_t (tf.Tensor): Coefficients calculated from the curve.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The scaled weights for kernel and bias.
        """
        if self.l2 is None:
            self.l2 = tf.Variable(0.0, trainable=False)
        weights = (
            self._compute_single_weights(self.curve_kernels, coeffs_t),
            self._compute_single_weights(self.curve_biases, coeffs_t),
        )
        return weights

    def _compute_single_weights(
        self, weights: List[tf.Variable], coeffs_t: tf.Tensor
    ) -> tf.Tensor:
        """Multiplies the given weights by the respective coefficient
        and adds them together.

        Adds to the l2 loss as well.

        Args:
            weights (Dict[int, tf.Variable]): The weights.
            coeffs_t (tf.Tensor): Coefficients calculated from the curve.

        Returns:
            tf.Tensor: The multiplied weights.
        """
        combined_weights = tf.stack([w.value() for w in weights], axis=-1)
        weights_avgeraged = tf.linalg.matvec(combined_weights, coeffs_t)

        # weight_sum = 0
        # for i in range(coeffs_t.shape[0]):
        #     weight_sum += weights[i].value() * coeffs_t[i]

        if weights_avgeraged is not None:
            # I think this should always be called
            self.l2.assign_add(tf.reduce_sum(weights_avgeraged**2))
        return weights_avgeraged


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

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        x, coeffs_t = inputs
        self.kernel, self.bias = self.compute_weights_t(coeffs_t)
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
        x, coeffs_t = inputs
        self.kernel, self.bias = self.compute_weights_t(coeffs_t)
        return tf.keras.layers.Dense.call(self, x)
