import logging
import re
from typing import Any, Dict, List, Type, Union

import numpy as np
import tensorflow as tf

from mode_connectivity.architecture import CurveModel
from mode_connectivity.curves import Curve
from mode_connectivity.layers import BatchNormalizationCurve, CurveLayer

logger = logging.getLogger(__name__)


class CurveNet(tf.keras.Model):
    num_classes: Union[int, None]
    num_bends: int
    fix_points: List[bool]
    curve: Curve
    curve_model: tf.keras.Model
    curve_layers: List[CurveLayer]

    def __init__(
        self,
        num_classes: Union[int, None],
        num_bends: int,
        weight_decay: float,  # TODO Add to architecture
        curve: Type[Curve],
        curve_model: Type[CurveModel],
        fix_start: bool = True,
        fix_end: bool = True,
        architecture_kwargs: Union[Dict[str, Any], None] = None,
    ):
        """Initializes the CurveNet consisiting of a CurveModel with CurveLayers (e.g. CNNCurve) and a Curve (e.g. Bezier).

        Args:
            num_classes (int): The amount of classes the net discriminates among.  Specified as "None" in regression tasks.
            num_bends (int): The amount of bends on the curve.
            weight_decay (float): Indicates the intensity of weight decay.
            curve (Type[Curve]): A parametric curve (e.g. Bezier).
            curve_model (Type[CurveModel]): The curve version of the utilized model (e.g. CNNCurve).
            fix_start (bool, optional): Boolean indicating whether the first bend/ pre-trained model on the curve is fixed. Defaults to True.
            fix_end (bool, optional): Boolean indicating whether the last bend/ pre-trained model on the curve is fixed. Defaults to True.
            architecture_kwargs (Union[Dict[str, Any], None], optional): Further arguments for the CurveModel. Defaults to None.

        Raises:
            ValueError: Indicates falsely specified curves.
        """
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
        """
        Imports parameters (kernels and biases) from the BaseModel into the CurveModel.

        Parameters in the BaseModel (without CurveLayers) are saved as
            <layerName>_<layerIndex>/<parameter>:0

            For example:
                conv2d/kernel:0
                conv2d/bias:0
                dense_1/kernel:0

        whereas parameters in the CurveModel are saved with a '_curve' suffix
        and an index that matches them to the bend/ point on curve.

            For example:
                conv2d_curve/kernel_curve_1:0
                conv2d_curve/bias_curve_0:0
                dense_1_curve/kernel_curve_2:0

        The parameters for a specific bend/ point on the curve are loaded from the pretrained BaseModel.

        Args:
            base_model (tf.keras.Model): The BaseModel (e.g. CNNBase) whose parameters are imported.
            index (int): Index of the bend/ point on curve.
        """
        if not self.curve_model.built:
            self._build_from_base_model(base_model)

        params = {p.name: p for p in self.curve_model.variables}
        base_params = {p.name: p for p in base_model.variables}

        assigned_params = []
        for name, param in params.items():
            parameter_index = self._find_parameter_index(name)
            if parameter_index != index:
                # Index of the Curve parameter (kernel, bias, ...) does not match
                # the specified index. Skip this parameter.
                continue

            base_name = self._get_base_name(name, index)
            base_param = base_params.get(base_name)
            if base_param is None:
                logger.debug(
                    f"Could not assign to param {name} (base_name: {base_name})"
                )
                continue

            param.assign(base_param.value())
            assigned_params.append(f"{base_name} -> {name}")

        logger.info(
            f"Assigned {len(assigned_params)} parameters for point #{index}: {', '.join(assigned_params)}"
        )

    def _build_from_base_model(self, base_model: tf.keras.Model):
        """
        Helper method that builds the CurveModel and thereby initializes its weights.
        It extracts information about the input shape from the corresponding BaseModel.

        Args:
            base_model (tf.keras.Model): The BaseModel (e.g. CNNBase).
        """
        base_input_shape = base_model.layers[0].input_shape
        point_on_curve_weights_input_shape = (len(self.fix_points),)
        input_shape = [
            tf.TensorShape(base_input_shape),
            tf.TensorShape(point_on_curve_weights_input_shape),
        ]
        self.curve_model.build(input_shape)

    @staticmethod
    def _find_parameter_index(parameter_name: str) -> Union[int, None]:
        """
        Finds the index of a bend/ curve point given a parameter name from the CurveModel.

        Args:
            parameter_name (str): Name of the parameter from the CurveModel.

        Returns:
            Union[int, None]: Index
        """
        results = re.findall(r"(\d+):", parameter_name)
        if not results:
            return None
        return int(results[0])

    @staticmethod
    def _get_base_name(parameter_name: str, index: int):
        """
        Returns the BaseModel parameter name, given a CurveModel parameter name.

        Args:
            parameter_name (str): Name of the parameter from the CurveModel.
            index (int): Index of the bend/ curve point.

        Returns:
            _type_: Name of the parameter in the BaseModel.
        """
        return parameter_name.replace("_curve", "").replace(f"_{index}:", ":")

    def init_linear(self) -> None:
        """
        Intitializes the inner bends/ points on the curve of each layer as a linear combination
        of the parameters (kernels and biases) of the first and last bend (pre-trained models).
        """
        for layer in self.curve_layers:
            for param_name in layer.parameters:
                self._compute_inner_params(parameters=layer.curve_params(param_name))

    def _compute_inner_params(self, parameters: List[tf.Variable]) -> None:
        """
        Helper method for init_linear that performs the initialization of the parameteres (kernel or bias) of each layer.

        Args:
            parameters (List[tf.Variable]): List of kernels or biases of a CurveLayer.
        """
        first_param, last_param = parameters[0].value(), parameters[-1].value()
        n_params = len(parameters)
        for i in range(1, n_params - 1):
            alpha = i * 1.0 / (n_params - 1)
            parameters[i].assign(alpha * first_param + (1.0 - alpha) * last_param)

    def get_weighted_parameters(self, point_on_curve):
        """
        TODO
        _summary_

        Args:
            point_on_curve (_type_): _description_

        Returns:
            _type_: _description_
        """
        point_on_curve_weights = self.curve(point_on_curve)
        parameters = []
        for layer in self.curve_layers:
            parameters.extend(
                [
                    w
                    for w in layer.compute_weighted_parameters(
                        point_on_curve_weights
                    ).values()
                    if w is not None
                ]
            )
        return np.concatenate([tf.stop_gradient(w).numpy().ravel() for w in parameters])

    @tf.function
    def generate_point_on_curve(self, dtype=tf.float32):
        """
        Samples a random point on the curve as a value in the range [0, 1) based  on the Uniform distribution.

        Args:
            dtype (_type_, optional): Defaults to tf.float32.

        Returns:
            _type_: Sampled point on curve.
        """
        return tf.random.uniform(shape=(), dtype=dtype)

    def call(self, inputs: tf.Tensor, training=None, update: bool = False):
        """
        Performs the forward pass of the CurveNet with input data.

        Args:
            inputs (tf.Tensor): Input data that is propagated through the CurveNet.
            training (_type_, optional): Boolean indicating train mode. Defaults to None.
            update (bool, optional): Wether to update special parameters. Used in the
                BatchNormalizationCurve Layer.

        Returns:
            _type_: Network predictions.
        """
        if training is not False:
            # If training is False, we are in evaluation.
            # The point_on_curve needs to be set beforehand, or is
            # generated by .evaluate_points()
            self.point_on_curve.assign(self.generate_point_on_curve(inputs.dtype))
        point_on_curve_weights = self.curve(self.point_on_curve)
        outputs = self.curve_model(
            (inputs, point_on_curve_weights), training=training, update=update
        )
        return outputs

    def evaluate_points(
        self,
        *args,
        num_points: Union[int, None] = None,
        point_on_curve: Union[float, None] = None,
        **kwargs,
    ):
        """
        Evaluates the CurveNet for one or several points on the curve.

        Args:
            num_points (Union[int, None], optional): Amount of equally spaced points on the curve to be evaluated. Defaults to None.
            point_on_curve (Union[float, None], optional): Single point on the curve to be evaluated specified by values in [0, 1]. Defaults to None.

        Raises:
            AttributeError: Indicates falsely specified attributes.

        Returns:
            _type_: _description_
        """
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
        positional_inputs = args[0] if args else None
        inputs = kwargs.get("x", positional_inputs)

        for point_on_curve in points_on_curve:
            print(f"{point_on_curve=:.3f}")
            self.point_on_curve.assign(
                tf.constant(point_on_curve, shape=(), dtype=tf.float32)
            )
            # Update moving mean/variance for BatchNorm Layers
            self.update_batchnorm(inputs=inputs)

            result = self.evaluate(*args, **kwargs)
            result["point_on_curve"] = point_on_curve
            results.append(result)
        return results

    @staticmethod
    def is_batchnorm(layer: tf.keras.layers.Layer) -> bool:
        return issubclass(
            layer.__class__,
            (tf.keras.layers.BatchNormalization, BatchNormalizationCurve),
        )

    @property
    def has_batchnorm(self) -> bool:
        return any(self.is_batchnorm(l) for l in self.layers)

    def update_batchnorm(self, inputs: tf.Tensor, **kwargs):
        if not self.has_batchnorm:
            logger.debug("Model has no BatchNormalisation Layer")
            return

        momenta = {}
        # Reset stats and save momentum for BatchNorm Layers
        for layer in self.layers:
            if self.is_batchnorm(layer):
                layer.reset_moving_stats()
                momenta[layer] = layer.momentum

        num_samples = 0
        # Run model over input batches to update moving mean/variance
        for input_ in inputs:
            if isinstance(input_, tuple):
                # If we are iterating over a dataloader
                input_ = input_[0]
            batch_size = len(input_)
            momentum = batch_size / (num_samples + batch_size)

            for layer in momenta.keys():
                layer.momentum = momentum

            self(input_, update=True, **kwargs)
            num_samples += batch_size

        for layer, momentum in momenta.items():
            layer.momentum = momentum
