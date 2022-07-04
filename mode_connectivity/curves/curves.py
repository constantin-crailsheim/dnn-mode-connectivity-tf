from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import tensorflow as tf
from scipy.special import binom


class Curve(tf.keras.layers.Layer, ABC):
    """Base class for Curve Layers.

    Args:
            num_bends (int): Degree of the Curve. Needs to be greater than 0.
                0: Linear
                1: Quadratic
                2: Cubic
                ...
    """

    num_bends: int

    def __init__(self, num_bends: int):
        """Initialize the Curve Layer."""
        super().__init__()
        if num_bends < 0:
            raise ValueError(
                f"Number of bends ({num_bends=}) of the curve needs to be greater or equal to 0."
            )
        self.num_bends = num_bends

    @abstractmethod
    def call(self, point_on_curve: Union[float, tf.Tensor]) -> tf.Tensor:
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}(Curve): num_bends={self.num_bends}>"


class Bezier(Curve):
    """Implementation of the Bezier Curve in a Layer context.

    Args:
            degree (int): Degree of the Curve. Needs to be greater than 0.
                1: Linear
                2: Quadratic
                3: Cubic
                ...

    Calling this layer returns a Tensor of weights, which sum up to 1.
    The weights correspond to the weight each input point of the Bezier curve is given when constructing the curve.
    This Tensor is then used to weigh parameters of CurveLayers.

    For further information, see:
    https://en.wikipedia.org/wiki/B%C3%A9zier_curve
    """

    def __init__(self, num_bends: int):
        super().__init__(num_bends=num_bends)
        self.degree = num_bends + 1
        self.binom = tf.Variable(
            tf.constant(
                binom(self.degree, np.arange(self.degree + 1), dtype=np.float32)
            ),
            trainable=False,
        )
        self.range = tf.Variable(tf.range(0, float(self.degree + 1)), trainable=False)
        self.rev_range = tf.Variable(
            tf.range(float(self.degree), -1, delta=-1), trainable=False
        )

    def call(self, point_on_curve: Union[float, tf.Tensor]) -> tf.Tensor:
        return (
            self.binom
            * tf.math.pow(point_on_curve, self.range)
            * tf.math.pow((1.0 - point_on_curve), self.rev_range)
        )

class PolyChain(Curve):
    def __init__(self, num_bends: int):
        super().__init__(num_bends=num_bends)
        self.num_bends = num_bends
        self.range = tf.Variable(tf.range(0, float(num_bends + 2)), trainable=False)

    def call(self, point_on_curve: Union[float, tf.Tensor]) -> tf.Tensor:
        t_n = point_on_curve * (self.num_bends + 1) # Better name for t_n
        tensor_of_zeros = tf.Variable(tf.zeros(self.num_bends + 2), trainable=False)
        point_on_curve_weight_tmp = 1.0 - tf.math.abs(t_n - self.range) # Find better name
        return (
            tf.math.maximum(tensor_of_zeros, point_on_curve_weight_tmp)
        )
