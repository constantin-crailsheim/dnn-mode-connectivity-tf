from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Union

import tensorflow as tf

from utils import split_list


# Proposal: Make this the parent class of all Curves (Bezier, Polychain, ...)
# This should be considered a Layer, since we directly compute in call(), right?
class Curve(tf.keras.layers.Layer, ABC):
    num_bends: int

    def __init__(self, num_bends: int):
        super().__init__()
        self.num_bends = num_bends

    @abstractmethod
    def call(self, uniform_tensor: tf.Tensor):
        pass


class CurveModule:
    pass


class CurveNet(tf.keras.Model):
    num_classes: int
    num_bends: int
    l2: float

    fix_points: List[bool]

    coeff_layer: Curve
    net: tf.keras.Model

    def __init__(
        self,
        num_classes: int,
        num_bends: int,
        curve: Type[Curve],  # Bezier, Polychain
        curve_model: Type[
            tf.keras.Model
        ],  # ConvFCCurve, VGGCurve -> Define CurveModel Type for Models of this kind?
        fix_start: bool = True,
        fix_end: bool = True,
        architecture_kwargs: Union[Dict[str, Any], None] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_bends = num_bends
        self.fix_points = [fix_start] + [False] * (self.num_bends - 2) + [fix_end]
        self.l2 = 0.0

        # TODO Rename class attributes as well for clarity?
        self.coeff_layer = curve(self.num_bends)
        self.net = curve_model(
            num_classes=num_classes, fix_points=self.fix_points, **architecture_kwargs
        )
        self.curve_modules = [
            module
            for module in self.net.submodules()  # TODO Need to check if this gathers the correct modules
            if issubclass(module.__class__, CurveModule)
        ]

    def import_base_parameters(self, base_model: tf.keras.Model, index: int) -> None:
        """Import parameters from the base model into this model.

        Args:
            base_model (tf.keras.Model): The base model from which parameters should be imported.
            index (int): _description_
        """
        variables: List[tf.Variable] = self.net.variables[index :: self.num_bends]
        base_variables: List[tf.Variable] = base_model.variables

        # or should we call get/set_weights ?
        # would be even better to initialize by names
        # the indexing [index :: self.num_bends] above is not so clear
        for variable, base_variable in zip(variables, base_variables):
            variable.assign(base_variable.value)

    def init_linear(self) -> None:
        """Initialize the linear layer of the model."""
        split_weights = split_list(self.net.variables, size=self.num_bends)
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
        self.l2 = sum(module.l2 for module in self.curve_modules)

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
        coeffs_t = self.coeff_layer(uniform_tensor)
        output = self.net(inputs, coeffs_t)
        self._compute_l2()
        return output

    def import_base_buffers(self, base_model: tf.keras.Model) -> None:
        # Not needed for now, only used in test_curve.py
        raise NotImplementedError

    def export_base_parameters(self, base_model: tf.keras.Model, index: int) -> None:
        # Not needed for now, actually never used in original repo
        raise NotImplementedError

    def weights(self, inputs: tf.Tensor):
        # Not needed for now, only called in eval_curve.py and plane.py
        raise NotImplementedError
