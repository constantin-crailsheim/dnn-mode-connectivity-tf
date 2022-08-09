from typing import List, Tuple

import tensorflow as tf
from mode_connectivity.architecture import Architecture, CurveModel
from mode_connectivity.layers import BatchNormalizationCurve, Conv2DCurve, DenseCurve

__all__ = [
    "CNN",
]


class CNNBase(tf.keras.Model):
    num_classes: int
    conv_part: tf.keras.Sequential
    fc_part: tf.keras.Sequential

    def __init__(self, num_classes: int, weight_decay: float):
        """
        Initializes the base version of the CNN.
        It consists of a convolutional and fully-connected part, each comprising several layers.
        The CNN performs image classification among several classes.

        Args:
            num_classes (int): The amount of classes the net discriminates among.
            weight_decay (float): Indicates the intensity of weight decay.
        """

        super().__init__()
        self.num_classes = num_classes
        regularizers = {
            "kernel_regularizer": tf.keras.regularizers.L2(weight_decay),
            "bias_regularizer": tf.keras.regularizers.L2(weight_decay),
        }

        self.conv_part = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(3, 3), activation="relu", **regularizers
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(3, 3), activation="relu", **regularizers
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), **regularizers),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
            ]
        )

        self.fc_part = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=64, activation="relu", **regularizers),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(units=64, activation="relu", **regularizers),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(units=self.num_classes, **regularizers),
            ]
        )

    def call(self, inputs: tf.Tensor, **kwargs):
        """
        Performs the forward pass of the base CNN with input data.

        Args:
            inputs (tf.Tensor): Input data that is propagated through the base CNN.

        Returns:
            tf.Tensor: Final layer output for each of the classes.
        """
        x = self.conv_part(inputs, **kwargs)
        return self.fc_part(x, **kwargs)


class CNNCurve(CurveModel):
    def __init__(self, num_classes: int, fix_points: List[bool], weight_decay: float):
        """
        Initializes the curve version of the CNN that consists of several Curve-Layers.
        The CNN performs image classification among several classes.

        Args:
            num_classes (int): The amount of classes the net discriminates among.
            fix_points (List[bool]): List of Booleans indicating for each bend/ point on curve if it is fixed. Defaults to True.
            weight_decay (float): Indicates the intensity of weight decay.
        """
        super().__init__(fix_points=fix_points)
        regularizers = {
            "kernel_regularizer": tf.keras.regularizers.L2(weight_decay),
            "bias_regularizer": tf.keras.regularizers.L2(weight_decay),
        }

        self.conv1 = Conv2DCurve(
            filters=32,
            kernel_size=(3, 3),
            fix_points=fix_points,
            activation="relu",
            **regularizers,
        )
        self.bn1 = BatchNormalizationCurve(fix_points=fix_points)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv2 = Conv2DCurve(
            filters=64,
            kernel_size=(3, 3),
            fix_points=fix_points,
            activation="relu",
            **regularizers,
        )
        self.bn2 = BatchNormalizationCurve(fix_points=fix_points)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv3 = Conv2DCurve(
            filters=64, kernel_size=(3, 3), fix_points=fix_points, **regularizers
        )
        self.bn3 = BatchNormalizationCurve(fix_points=fix_points)
        self.flatten1 = tf.keras.layers.Flatten()
        self.conv_part = [
            self.conv1,
            self.bn1,
            self.pool1,
            self.conv2,
            self.bn2,
            self.pool2,
            self.conv3,
            self.bn3,
            self.flatten1,
        ]

        self.dense1 = DenseCurve(
            units=64, fix_points=fix_points, activation="relu", **regularizers
        )
        self.bn4 = BatchNormalizationCurve(fix_points=fix_points)
        self.dense2 = DenseCurve(
            units=64, fix_points=fix_points, activation="relu", **regularizers
        )
        self.bn5 = BatchNormalizationCurve(fix_points=fix_points)
        self.dense3 = DenseCurve(
            units=num_classes, fix_points=fix_points, **regularizers
        )
        self.fc_part = [self.dense1, self.bn4, self.dense2, self.bn5, self.dense3]

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], **kwargs):
        """
        Performs the forward pass of the curve CNN with input data.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): Input data that is propagated through the curve CNN with bend weights defining the point on curve.

        Returns:
            tf.Tensor: Final layer output for each of the classes.
        """
        x, point_on_curve_weights = inputs

        x = self.conv1((x, point_on_curve_weights))
        x = self.bn1((x, point_on_curve_weights), **kwargs)
        x = self.pool1(x)
        x = self.conv2((x, point_on_curve_weights))
        x = self.bn2((x, point_on_curve_weights), **kwargs)
        x = self.pool2(x)
        x = self.conv3((x, point_on_curve_weights))
        x = self.bn3((x, point_on_curve_weights), **kwargs)
        x = self.flatten1(x)

        x = self.dense1((x, point_on_curve_weights))
        x = self.bn4((x, point_on_curve_weights), **kwargs)
        x = self.dense2((x, point_on_curve_weights))
        x = self.bn5((x, point_on_curve_weights), **kwargs)
        x = self.dense3((x, point_on_curve_weights))
        return x


class CNN(Architecture):
    base = CNNBase
    curve = CNNCurve
    kwargs = {}
