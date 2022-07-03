from typing import List, Tuple

import tensorflow as tf

from mode_connectivity.curves.layers import Conv2DCurve, DenseCurve

__all__ = [
    "CNN",
]


class CNNBase(tf.keras.Model):
    num_classes: int
    conv_part: tf.keras.Sequential
    fc_part: tf.keras.Sequential

    # Sources:
    # https://www.tensorflow.org/tutorials/customization/custom_layers
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers

    # Network Structure:
    # https://github.com/constantin-crailsheim/dnn-mode-connectivity/blob/master/models/basiccnn.py

    # Optional: Alternatively inherit from tf.keras.layers.Layer
    # https://stackoverflow.com/questions/55109696/tensorflow-difference-between-tf-keras-layers-layer-vs-tf-keras-model

    # Comment: In contrast to PyTorch there are no input dimensions required in Tensorflow.

    def __init__(self, num_classes: int, weight_decay: float):
        super().__init__()
        self.num_classes = num_classes

        self.conv_part = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(3, 3),
                    activation="relu",
                    kernel_initializer="glorot_normal",
                    bias_initializer="zeros",
                    kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                ),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    activation="relu",
                    kernel_initializer="glorot_normal",
                    bias_initializer="zeros",
                    kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                ),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    kernel_initializer="glorot_normal",
                    bias_initializer="zeros",
                    kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                ),
                tf.keras.layers.Flatten(),
            ]
        )

        self.fc_part = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=64,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                ),
                tf.keras.layers.Dense(
                    units=64,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                ),
                tf.keras.layers.Dense(
                    units=self.num_classes,
                    kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                ),
            ]
        )  # Check if weight decay needed in each layer

    def call(
        self, inputs, training=None, mask=None
    ):  # TO DO: Typehints & Check which arguments necessary
        x = self.conv_part(inputs)
        return self.fc_part(x)
        # TODO: Check if x = x.view(x.size(0), -1) necessary


class CNNCurve(tf.keras.Model):
    def __init__(self, num_classes: int, fix_points: List[bool], weight_decay: float):
        super().__init__()

        self.conv1 = Conv2DCurve(
            filters=32,
            kernel_size=(3, 3),
            fix_points=fix_points,
            activation="relu",
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv2 = Conv2DCurve(
            filters=64,
            kernel_size=(3, 3),
            fix_points=fix_points,
            activation="relu",
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv3 = Conv2DCurve(
            filters=64,
            kernel_size=(3, 3),
            fix_points=fix_points,
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        )
        self.flatten1 = tf.keras.layers.Flatten()
        self.conv_part = [
            self.conv1,
            self.pool1,
            self.conv2,
            self.pool2,
            self.conv3,
            self.flatten1,
        ]

        self.dense1 = DenseCurve(
            units=64,
            fix_points=fix_points,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        )
        self.dense2 = DenseCurve(
            units=64,
            fix_points=fix_points,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        )
        self.dense3 = DenseCurve(
            units=num_classes,
            fix_points=fix_points,
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        )
        self.fc_part = [self.dense1, self.dense2, self.dense3]

    def call(
        self, inputs: Tuple[tf.Tensor, tf.Tensor], training=None, mask=None
    ):  # TO DO: Typehints
        x, point_on_curve_weights = inputs

        x = self.conv1((x, point_on_curve_weights))
        x = self.pool1(x)
        x = self.conv2((x, point_on_curve_weights))
        x = self.pool2(x)
        x = self.conv3((x, point_on_curve_weights))
        x = self.flatten1(x)

        x = self.dense1((x, point_on_curve_weights))
        x = self.dense2((x, point_on_curve_weights))
        x = self.dense3((x, point_on_curve_weights))
        # TODO: Check if x = x.view(x.size(0), -1) necessary

        return x


class CNN:
    base = CNNBase
    curve = CNNCurve
    kwargs = {}
