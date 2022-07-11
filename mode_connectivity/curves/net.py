import logging
import re
from typing import Any, Dict, List, Type, Union

import tensorflow as tf
import numpy as np

from mode_connectivity.curves.curves import Curve
from mode_connectivity.curves.layers import CurveLayer

logger = logging.getLogger(__name__)


class CurveNet(tf.keras.Model):
    num_classes: int
    num_bends: int

    fix_points: List[bool]

    curve: Curve
    curve_model: tf.keras.Model
    curve_layers: List[CurveLayer]

    def __init__(
        self,
        num_classes: int,
        num_bends: int,
        weight_decay: float, # TODO Add to architecture
        curve: Type[Curve],  # Bezier, Polychain
        curve_model: Type[tf.keras.Model],  # ConvFCCurve, VGGCurve
        fix_start: bool = True,
        fix_end: bool = True,
        architecture_kwargs: Union[Dict[str, Any], None] = None,
    ):
        super().__init__()
        if num_bends < 0:
            raise ValueError(
                f"Number of bends of the curve need to be at least 0 (found {num_bends=})."
            )
        if num_bends == 0 and fix_start and fix_end:
            logger.warning(
                "You specified no bends for the curve, but fixed both start and end point. "
                "Training this model will give no results if all weights are fixed!"
            )
        self.num_classes = num_classes
        self.num_bends = num_bends
        self.point_on_curve = tf.Variable(0.0, trainable=False, name="point_on_curve")
        self.fix_points = [fix_start] + [False] * self.num_bends + [fix_end]

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
                logger.debug(f"Index of {name} does not match kernel/bias index.")
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
            f"Assigned {len(assigned_weights)} weights for point #{index}: {', '.join(assigned_weights)}"
        )

    def _build_from_base_model(self, base_model: tf.keras.Model):
        """Build the curve model to initialize weights."""
        base_input_shape = base_model.layers[0].input_shape
        point_on_curve_weights_input_shape = (len(self.fix_points),)
        input_shape = [
            tf.TensorShape(base_input_shape),
            tf.TensorShape(point_on_curve_weights_input_shape),
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

    def get_weighted_parameters(self, point_on_curve):
        point_on_curve_weights = self.curve(point_on_curve)
        parameters = []
        for module in self.curve_layers:
            parameters.extend(
                [
                    w
                    for w in module.compute_weighted_parameters(point_on_curve_weights)
                    if w is not None
                ]
            )
        return np.concatenate(
            [tf.stop_gradient(w).numpy().ravel() for w in parameters]
        )  # .cpu() missing

    def _compute_inner_weights(self, weights: List[tf.Variable]) -> None:
        # Is this procedure mentioned somewhere in the paper?
        first_weight, last_weight = weights[0].value(), weights[-1].value()
        n_weights = len(weights)
        for i in range(1, n_weights - 1):
            alpha = i * 1.0 / (n_weights - 1)
            weights[i].assign(alpha * first_weight + (1.0 - alpha) * last_weight)

    @tf.function
    def generate_point_on_curve(self, dtype=tf.float32):
        return tf.random.uniform(shape=(), dtype=dtype)

    def call(
        self,
        inputs: tf.Tensor,
        training=None, # Why are we not setting this to True as default
        mask=None, # For what do we need the mask?
    ):
        if training is not False:
            # If training is False, we are in evaluation.
            # The point_on_curve needs to be set beforehand, or is
            # generated by .evaluate()
            self.point_on_curve.assign(self.generate_point_on_curve(inputs.dtype))
        point_on_curve_weights = self.curve(self.point_on_curve)
        outputs = self.curve_model((inputs, point_on_curve_weights))
        return outputs

    def evaluate_points(
        self,
        *args,
        num_points: Union[int, None] = None,
        point_on_curve: Union[float, None] = None,
        **kwargs,
    ):
        if not (num_points is None or point_on_curve is None):
            raise AttributeError(
                "Cannot specify both 'num_points' and 'point_on_curve'. "
                f"Got values {num_points=}, {point_on_curve=}"
            )
        if num_points is None and point_on_curve is None:
            raise AttributeError(
                "Need to specify one of 'num_points' or 'point_on_curve'. "
                f"Got values {num_points=}, {point_on_curve=}"
            )

        points_on_curve = []
        if point_on_curve is not None:
            points_on_curve.append(point_on_curve)
        if num_points is not None:
            points_on_curve += list(np.linspace(0.0, 1.0, num_points))

        print(f"Evaluating CurveNet for {[f'{p:.3f}' for p in points_on_curve]}")
        results = []
        for point_on_curve in points_on_curve:
            print(f"{point_on_curve=:.3f}")
            self.point_on_curve.assign(
                tf.constant(point_on_curve, shape=(), dtype=tf.float32)
            )
            result = super().evaluate(*args, **kwargs)
            result["point_on_curve"] = point_on_curve
            results.append(result)
        return results

    def import_base_buffers(self, base_model: tf.keras.Model) -> None:
        # Not needed for now, only used in test_curve.py
        raise NotImplementedError()

    def export_base_parameters(self, base_model: tf.keras.Model, index: int) -> None:
        # Not needed for now, actually never used in original repo
        raise NotImplementedError()

    # def weights(self, inputs: tf.Tensor):
    #     # Not needed for now, only called in eval_curve.py and plane.py
    #     raise NotImplementedError()
