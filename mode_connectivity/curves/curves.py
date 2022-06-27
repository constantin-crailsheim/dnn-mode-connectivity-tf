from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import tensorflow as tf
from scipy.special import binom


class Curve(tf.keras.layers.Layer, ABC):
    num_bends: int

    def __init__(self, num_bends: int):
        super().__init__()

        if num_bends < 1:
            raise ValueError(
                f"Number of bends ({num_bends=}) needs to be greater than 0."
            )
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

    def call(self, curve_point: Union[float, tf.Tensor]) -> tf.Tensor:
        return (
            self.binom
            * tf.math.pow(curve_point, self.range)
            * tf.math.pow((1.0 - curve_point), self.rev_range)
        )
