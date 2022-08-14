from typing import List, Tuple

import tensorflow as tf
from mode_connectivity.architecture import Architecture, CurveModel
from mode_connectivity.layers import DenseCurve

__all__ = [
    "MLP",
]


class MLPBase(tf.keras.Model):
    def __init__(self, num_classes: None, weight_decay: float):
        """
        Initializes the base version of the MLP.
        It consists of a fully-connected part comprising several layers.
        The MLP regresses the target variable on the independent variables.

        Args:
            num_classes (None): Specified as "None" in this regression tasks.
            weight_decay (float): Indicates the intensity of weight decay.
        """
        super().__init__()
        regularizers = {
            "kernel_regularizer": tf.keras.regularizers.L2(weight_decay),
            "bias_regularizer": tf.keras.regularizers.L2(weight_decay),
        }
        self.fc_part = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(units=16, activation="tanh", **regularizers),
                tf.keras.layers.Dense(units=1, activation="linear", **regularizers),
            ]
        )

    def call(self, inputs: tf.Tensor, **kwargs):
        """
        Performs the forward pass of the base MLP with input data.

        Args:
            inputs (tf.Tensor): Input data that is propagated through the base MLP.

        Returns:
            _type_: Network predictions.
        """
        return self.fc_part(inputs, **kwargs)


class MLPCurve(CurveModel):
    def __init__(self, num_classes: None, fix_points: List[bool], weight_decay: float):
        """
        Initializes the curve version of the MLP.
        It consists of a fully-connected part comprising several Curve-Layers.
        The MLP regresses the target variable on the independent variables.

        Args:
            num_classes (None): Specified as "None" in this regression tasks.
            fix_points (List[bool]): List of Booleans indicating for each bend/ point on curve if it is fixed. Defaults to True.
            weight_decay (float): Indicates the intensity of weight decay.
        """
        super().__init__(fix_points=fix_points)
        regularizers = {
            "kernel_regularizer": tf.keras.regularizers.L2(weight_decay),
            "bias_regularizer": tf.keras.regularizers.L2(weight_decay),
        }

        self.dense1 = DenseCurve(
            units=16, fix_points=fix_points, activation="tanh", **regularizers
        )
        self.dense2 = DenseCurve(
            units=1, fix_points=fix_points, activation="linear", **regularizers
        )

        self.fc_part = [self.dense1, self.dense2]

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training=None, mask=None):
        """
        Performs the forward pass of the curve MLP with input data.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]):  Input data that is propagated through the curve MLP with bend weights defining the point on curve.
            training (_type_, optional): Unused?. Defaults to None.
            mask (_type_, optional): Unused?. Defaults to None.

        Returns:
            tf.Tensor: Final layer output/ Prediction of MLP.
        """
        x, point_on_curve_weights = inputs

        x = self.dense1((x, point_on_curve_weights))
        x = self.dense2((x, point_on_curve_weights))

        return x


class MLP(Architecture):
    base = MLPBase
    curve = MLPCurve
    kwargs = {}
