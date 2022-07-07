from typing import List, Tuple

import tensorflow as tf

from mode_connectivity.curves.layers import DenseCurve

__all__ = [
    "MLP",
]


class MLPBase(tf.keras.Model):  # Inherit equivalent of torch.nn
    def __init__(self, num_classes: int, weight_decay: float):
        super(MLPBase, self).__init__()
        self.fc_part = tf.keras.models.Sequential(
            [
                # tf.keras.layers.Dense(
                #     units=16,
                #     activation="tanh",
                #     kernel_regularizer = tf.keras.regularizers.L2(weight_decay)),
                tf.keras.layers.Dense(
                    units=4,
                    activation="tanh",
                    kernel_regularizer = tf.keras.regularizers.L2(weight_decay)),
                tf.keras.layers.Dense(
                    units=1,
                    activation="linear",
                    kernel_regularizer = tf.keras.regularizers.L2(weight_decay)),
            ]
        )

    def call(self, x):
        x = self.fc_part(x)
        return x


class MLPCurve(tf.keras.Model):  # Inherit equivalent of torch.nn
    def __init__(self, num_classes: int, fix_points: List[bool], weight_decay: float):
        super(MLPCurve, self).__init__()

        # self.dense1 = DenseCurve(
        #     units=16,
        #     fix_points=fix_points,
        #     activation="relu",
        #     kernel_regularizer = tf.keras.regularizers.L2(weight_decay)
        # )
        self.dense1 = DenseCurve(
            units=4,
            fix_points=fix_points,
            activation="tanh",
            kernel_regularizer = tf.keras.regularizers.L2(weight_decay)
        )
        self.dense2 = DenseCurve(
            units=1,
            fix_points=fix_points,
            activation="linear",
            kernel_regularizer = tf.keras.regularizers.L2(weight_decay)
        )

        self.fc_part = [self.dense1, self.dense2] # , self.dense3

    def call(
        self, inputs: Tuple[tf.Tensor, tf.Tensor], training=None, mask=None
    ):
        x, point_on_curve_weights = inputs

        x = self.dense1((x, point_on_curve_weights))
        x = self.dense2((x, point_on_curve_weights))
        # x = self.dense3((x, point_on_curve_weights))

        return x


class MLP:
    base = MLPBase
    curve = MLPCurve
    kwargs = {}
