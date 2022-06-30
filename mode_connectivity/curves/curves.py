from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import tensorflow as tf
from scipy.special import binom


class Curve(tf.keras.layers.Layer, ABC):
    """Base class for Curve Layers.

    Args:
            degree (int): Degree of the Curve. Needs to be greater than 0.
                1: Linear
                2: Quadratic
                3: Cubic
                ...
    """

    degree: int

    def __init__(self, degree: int):
        """Initialize the Curve Layer."""
        super().__init__()
        if degree < 1:
            raise ValueError(
                f"Degree ({degree=}) of the curve needs to be greater than 0."
            )
        self.degree = degree

    @abstractmethod
    def call(self, point_on_curve: Union[float, tf.Tensor]) -> tf.Tensor:
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}(Curve): degree={self.degree}>"


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

    def __init__(self, degree: int):
        super().__init__(degree=degree)
        self.binom = tf.Variable(
            tf.constant(binom(degree, np.arange(degree + 1), dtype=np.float32)),
            trainable=False,
        )
        self.range = tf.Variable(tf.range(0, float(degree + 1)), trainable=False)
        self.rev_range = tf.Variable(
            tf.range(float(degree), -1, delta=-1), trainable=False
        )

    def call(self, point_on_curve: Union[float, tf.Tensor]) -> tf.Tensor:
        return (
            self.binom
            * tf.math.pow(point_on_curve, self.range)
            * tf.math.pow((1.0 - point_on_curve), self.rev_range)
        )
