from abc import ABC
from typing import Dict, List, Tuple, Type, Union

import tensorflow as tf


class CurveLayer(tf.keras.layers.Layer, ABC):
    fix_points: List[bool]
    num_bends: int
    base_layer: Type[tf.keras.layers.Layer]
    parameters: Tuple[str]

    def __init__(
        self,
        fix_points: List[bool],
        base_layer: Type[tf.keras.layers.Layer],
        parameters: Tuple[str] = ("kernel", "bias"),
        **kwargs,
    ):
        if len(fix_points) < 2:
            raise ValueError(
                f"You need to specify at least two points (found {len(fix_points)})!"
            )

        super().__init__(**kwargs)
        self.base_layer = base_layer
        self.fix_points = fix_points
        self.num_bends = len(self.fix_points) - 2
        self.parameters = parameters
        self._reset_input_spec()

    def _reset_input_spec(self):
        """Modify the input specification to take in the curve coefficients as well."""
        self.input_spec = [
            self.input_spec,
            tf.keras.layers.InputSpec(shape=((len(self.fix_points),))),
        ]

    def build(self, input_shape):
        """Build the curve layer and register curve parameters.

        Args:
            input_shape: Shape of inputs

        Workaround to use the base_layers build() method, explained for
        parameters = ('kernel', 'bias'):
            We need to remove the kernel and bias regularizer temporarly for building
            or otherwise the kernel/bias-regularizer will get registered and thus
            taken into account when calculating the loss.
            regularizers = [getattr(self, f"{param}_regularizer") for param in self.parameters]
            kernel_reg, bias_reg = self.kernel_regularizer, self.bias_regularizer
        """
        # Remove and store parameter regularizers
        regularizers = {}
        for param_name in self.parameters:
            reg_name = f"{param_name}_regularizer"
            regularizers[reg_name] = getattr(self, reg_name)
            setattr(self, reg_name, None)

        # Build the layer
        self.base_layer.build(self, input_shape[0])
        self._reset_input_spec()

        # Restore parameter regularizers
        for reg_name, regularizer in regularizers.items():
            setattr(self, reg_name, regularizer)

        # Register curve parameters (e.g. curve_kernels, curve_biases)
        self.add_parameter_weights()
        # Delete old paramters, so they are not registered as
        # trainable variables
        for param_name in self.parameters:
            delattr(self, param_name)

    # TODO Check inputs as Tuple or seperate
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        x, curve_point_weights = inputs
        self.compute_and_set_weighted_parameters(curve_point_weights)
        return self.base_layer.call(self, x)

    def add_parameter_weights(self):
        """Add parameter weights for each curve point.

        This method needs to be called in the build() method of
        the new layer.

        Variables for each paramter are saved in lists in
        as class attributes called `curve_<parameter_name>s`, with
        the index of the curve point as the respective key.
        E.g. for self.parameters = ('kernel', 'bias'):
        self.curve_kernels and self.curve_biases
        """
        for param_name in self.parameters:
            curve_params = [None] * len(self.fix_points)
            for i, fixed in enumerate(self.fix_points):
                if getattr(self, param_name) is not None:
                    curve_params[i] = self._add_parameter(
                        param_name=param_name, index=i, fixed=fixed
                    )
            setattr(self, self._get_curve_param_name(param_name), curve_params)

    @staticmethod
    def _get_curve_param_name(param_name: str) -> str:
        return f"curve_{param_name}{'es' if param_name[-1] == 's' else 's'}"

    def curve_params(self, param_name: str) -> List[tf.Variable]:
        return getattr(self, self._get_curve_param_name(param_name))

    def _add_parameter(self, param_name: str, index: int, fixed: bool):
        name = f"{param_name}_curve_{index}"
        weight = self.add_weight(
            name=name,
            shape=getattr(self, param_name).shape,
            initializer=getattr(self, f"{param_name}_initializer"),
            regularizer=getattr(self, f"{param_name}_regularizer"),
            constraint=getattr(self, f"{param_name}_constraint"),
            trainable=not fixed,
            dtype=self.dtype,
        )
        return weight

    def compute_and_set_weighted_parameters(
        self, curve_point_weights: tf.Tensor
    ) -> None:
        computed_parameters = self.compute_weighted_parameters(curve_point_weights)
        for param_name, param in computed_parameters.items():
            setattr(self, param_name, param)

    def compute_weighted_parameters(
        self, curve_point_weights: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Compute combined weights for each parameter.

        Args:
            curve_point_weights (tf.Tensor): Coefficients calculated from the curve.

        Returns:
            Dict[str, tf.Tensor]: The scaled weights for each parameter.
        """
        computed_params = {}
        for param_name in self.parameters:
            curve_params = self.curve_params(param_name)
            computed_params[param_name] = self._compute_single_parameter(
                curve_params, curve_point_weights
            )
        return computed_params

    def _compute_single_parameter(
        self, parameters: List[tf.Variable], curve_point_weights: tf.Tensor
    ) -> tf.Tensor:
        """Multiplies the given weights by the respective coefficient
        and adds them together.

        Args:
            weights (Dict[int, tf.Variable]): The weights.
            curve_point_weights (tf.Tensor): Coefficients calculated from the curve.

        Returns:
            tf.Tensor: The multiplied weights.
        """
        combined_params = tf.stack([w.value() for w in parameters], axis=-1)
        params_averaged = tf.linalg.matvec(combined_params, curve_point_weights)
        return params_averaged


class Conv2DCurve(CurveLayer, tf.keras.layers.Conv2D):
    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        fix_points: List[bool],
        **kwargs,
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            fix_points=fix_points,
            base_layer=tf.keras.layers.Conv2D,
            parameters=("kernel", "bias"),
            **kwargs,
        )


class DenseCurve(CurveLayer, tf.keras.layers.Dense):
    def __init__(
        self,
        units: int,
        fix_points: List[bool],
        **kwargs,
    ):
        super().__init__(
            units=units,
            fix_points=fix_points,
            base_layer=tf.keras.layers.Dense,
            parameters=("kernel", "bias"),
            **kwargs,
        )


class BatchNormalizationCurve(CurveLayer, tf.keras.layers.BatchNormalization):
    def __init__(
        self,
        fix_points: List[bool],
        **kwargs,
    ):
        super().__init__(
            fix_points=fix_points,
            base_layer=tf.keras.layers.BatchNormalization,
            parameters=("gamma", "beta"),
            **kwargs,
        )
