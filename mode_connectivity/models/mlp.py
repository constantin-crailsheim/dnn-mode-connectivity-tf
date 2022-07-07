from typing import List, Tuple

import tensorflow as tf
from mode_connectivity.curves.layers import DenseCurve

__all__ = [
    "MLP",
]


class MLPBase(tf.keras.Model):
    def __init__(self, num_classes: int, weight_decay: float):
        super().__init__()
        regularizers = {
            "kernel_regularizer": tf.keras.regularizers.L2(weight_decay),
            "bias_regularizer": tf.keras.regularizers.L2(weight_decay),
        }
        self.fc_part = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(units=16, activation="relu", **regularizers),
                tf.keras.layers.Dense(units=8, activation="relu", **regularizers),
                tf.keras.layers.Dense(units=1, activation="linear", **regularizers),
            ]
        )

    def call(self, inputs: tf.Tensor, **kwargs):
        return self.fc_part(inputs, **kwargs)


class MLPCurve(tf.keras.Model):
    def __init__(self, num_classes: int, fix_points: List[bool], weight_decay: float):
        super().__init__()
        regularizers = {
            "kernel_regularizer": tf.keras.regularizers.L2(weight_decay),
            "bias_regularizer": tf.keras.regularizers.L2(weight_decay),
        }

        self.dense1 = DenseCurve(
            units=16, fix_points=fix_points, activation="relu", **regularizers
        )
        self.dense2 = DenseCurve(
            units=8, fix_points=fix_points, activation="relu", **regularizers
        )
        self.dense3 = DenseCurve(
            units=1, fix_points=fix_points, activation="linear", **regularizers
        )

        self.fc_part = [self.dense1, self.dense2, self.dense3]

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], **kwargs):
        x, coeffs_t = inputs

        x = self.dense1((x, coeffs_t), **kwargs)
        x = self.dense2((x, coeffs_t), **kwargs)
        x = self.dense3((x, coeffs_t), **kwargs)

        return x


class MLP:
    base = MLPBase
    curve = MLPCurve
    kwargs = {}
