from abc import ABC
from typing import Dict, List, Tuple, Type, Union

import tensorflow as tf


class CurveLayer(tf.keras.layers.Layer, ABC):
    """
    Base class for the Curve implementations of TensorFlow-Layers.
    A Curve consists of several nodes that are represented by a list of parameters (kernels and biases).
    For each of those nodes it can be specified if it is trainable or not/ fixed.
    """

    fix_points: List[bool]
    num_bends: int
    base_layer: Type[tf.keras.layers.Layer]
    parameter_types: Tuple[str]

    def __init__(
        self,
        fix_points: List[bool],
        base_layer: Type[tf.keras.layers.Layer],
        parameter_types: Tuple[str] = ("kernel", "bias"),
        **kwargs,
    ): 
        """
        Initializes the CurveLayer.

        Args:
            fix_points (List[bool]): List indicating for each node on the curve whether it is fixed/ not trainable.
            base_layer (Type[tf.keras.layers.Layer]): Corresponding  TensorFlow-Layer.
            parameter_types (Tuple[str], optional): List of parameter types. Defaults to ("kernel", "bias").

        Raises:
            ValueError: Indicates if the amount of nodes is misspecified.
        """
        if len(fix_points) < 2:
            raise ValueError(
                f"You need to specify at least two nodes (found {len(fix_points)})!"
            )

        super().__init__(**kwargs)
        self.base_layer = base_layer
        self.fix_points = fix_points
        self.num_bends = len(self.fix_points) - 2
        self.parameter_types = parameter_types
        self._reset_input_spec()

    def _reset_input_spec(self):
        """Modifies the input shape specification to account for the nodes."""
        self.input_spec = [
            self.input_spec,
            tf.keras.layers.InputSpec(shape=((len(self.fix_points),))),
        ]

    def build(self, input_shape):
        """
        Builds the curve layer and registers its parameters.

        Args:
            input_shape: Shape of inputs

        Workaround to use the base_layers build() method.
        Explained for parameter_types = ('kernel', 'bias'):
            We need to remove the kernel and bias regularizer temporarly for building
            or otherwise the kernel/bias-regularizer will get registered and thus
            taken into account when calculating the loss.
            regularizers = [getattr(self, f"{param}_regularizer") for param in self.parameters]
            kernel_reg, bias_reg = self.kernel_regularizer, self.bias_regularizer
        """
        # Remove and store parameter regularizers
        regularizers = {}
        for param_type in self.parameter_types:
            reg_name = f"{param_type}_regularizer"
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
        for param_type in self.parameter_types:
            delattr(self, param_type)
            
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], *args, **kwargs):
        """
        Applies the CurveLayer to inputs.
        As parameters it uses a weighted sum of the parameters of the CurveLayer nodes.
        This corresponds to the model lying on the sampled point on the curve.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): Tuple of layer inputs and weights of the nodes.

        Returns:
            _type_: Layer output
        """
        x, curve_point_weights = inputs
        self.compute_and_set_weighted_parameters(curve_point_weights)
        return self.base_layer.call(self, x, *args, **kwargs)

    def add_parameter_weights(self):
        """
        Adds the parameters for all parameter types and all curve nodes.
        Called in the build() method of the CurveLayer.

        Variables for each paramter type are saved in lists 
        as class attributes called `curve_<parameter_type>s`, with
        the index of the curve node as the respective key.
        E.g. for self.parameter_types = ('kernel', 'bias'):
        self.curve_kernels and self.curve_biases
        """
        for param_type in self.parameter_types:
            curve_params = [None] * len(self.fix_points)
            for i, fixed in enumerate(self.fix_points):
                if getattr(self, param_type) is not None:
                    curve_params[i] = self._add_parameter(
                        param_type=param_type, index=i, fixed=fixed
                    )
            setattr(self, self._get_curve_param_type(param_type), curve_params)

    @staticmethod
    def _get_curve_param_type(param_type: str) -> str:
        """
        Returns the CurveModel parameter type given a BaseModel parameter type.
        e.g. "curve_biases" for "bias"

        Args:
            param_type (str): BaseModel parameter type.

        Returns:
            str: CurveModel parameter type.
        """
        return f"curve_{param_type}{'es' if param_type[-1] == 's' else 's'}"

    def curve_params(self, param_type: str) -> List[tf.Variable]:
        """
        Returns the instance of a specified CurveModel parameter type.
        e.g. curve_biases-Instance for "bias"

        Args:
            param_type (str): BaseModel parameter type.

        Returns:
            List[tf.Variable]: CurveModel parameter.
        """
        return getattr(self, self._get_curve_param_type(param_type))

    def _add_parameter(self, param_type: str, index: int, fixed: bool):
        """
        Adds a node parameter of specified type to the CurveLayer.
        e.g. bias_curve_0

        Args:
            param_type (str): BaseModel parameter type.
            index (int): Index of the node.
            fixed (bool): Indicates if the node is fixed/ not trainable.

        Returns:
            _type_: Node parameter.
        """
        name = f"{param_type}_curve_{index}"
        parameter = self.add_weight(
            name=name,
            shape=getattr(self, param_type).shape,
            initializer=getattr(self, f"{param_type}_initializer"),
            regularizer=getattr(self, f"{param_type}_regularizer"),
            constraint=getattr(self, f"{param_type}_constraint"),
            trainable=not fixed,
            dtype=self.dtype,
        )
        return parameter

    def compute_and_set_weighted_parameters(
        self, curve_point_weights: tf.Tensor
    ) -> None:
        """
        Triggers the computation of weighted parameters and sets them.

        Args:
            curve_point_weights (tf.Tensor): Node weights leading to a certain point on curve.
        """
        computed_parameters = self.compute_weighted_parameters(curve_point_weights)
        for param_type, param in computed_parameters.items():
            setattr(self, param_type, param)

    def compute_weighted_parameters(
        self, curve_point_weights: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Computes weighted parameters based on the weights of each node for a each of the parameters types.

        Args:
            curve_point_weights (tf.Tensor): Node weights leading to a certain point on curve.

        Returns:
            Dict[str, tf.Tensor]: Weighted parameters.
        """
        computed_params = {}
        for param_type in self.parameter_types:
            curve_params = self.curve_params(param_type)
            computed_params[param_type] = self._compute_single_parameter(
                curve_params, curve_point_weights
            )
        return computed_params

    def _compute_single_parameter(
        self, parameters: List[tf.Variable], curve_point_weights: tf.Tensor
    ) -> tf.Tensor:
        """
        For a certain parameter type multiplies the parameters of the nodes by the respective weights and adds them up.

        Args:
            parameters (List[tf.Variable]): Parameters of the nodes.
            curve_point_weights (tf.Tensor): Weights of the nodes.

        Returns:
            tf.Tensor: Reweighted parameter.
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
        """
        Initializes the CurveLayer-Implementation of the Conv2D-Layer.

        Args:
            filters (int): Dimensionality of the output space.
            kernel_size (Union[int, Tuple[int, int]]): Size of the 2D convolution window.
            fix_points (List[bool]): List indicating for each node on the curve whether it is fixed/ not trainable.
        """
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            fix_points=fix_points,
            base_layer=tf.keras.layers.Conv2D,
            parameter_types=("kernel", "bias"),
            **kwargs,
        )

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        return super().call(inputs)


class DenseCurve(CurveLayer, tf.keras.layers.Dense):
    def __init__(
        self,
        units: int,
        fix_points: List[bool],
        **kwargs,
    ):
        """
        Initializes the CurveLayer-Implementation of the Dense-Layer.

        Args:
            units (int): Dimensionality of the output space.
            fix_points (List[bool]): List indicating for each node on the curve whether it is fixed/ not trainable.
        """
        super().__init__(
            units=units,
            fix_points=fix_points,
            base_layer=tf.keras.layers.Dense,
            parameter_types=("kernel", "bias"),
            **kwargs,
        )

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        return super().call(inputs)


class BatchNormalizationCurve(CurveLayer, tf.keras.layers.BatchNormalization):
    def __init__(
        self,
        fix_points: List[bool],
        **kwargs,
    ):
        """
        Initializes the CurveLayer-Implementation of the BatchNormalization-Layer.

        Args:
            fix_points (List[bool]): List indicating for each node on the curve whether it is fixed/ not trainable.
        """
        super().__init__(
            fix_points=fix_points,
            base_layer=tf.keras.layers.BatchNormalization,
            parameter_types=("gamma", "beta"),
            **kwargs,
        )

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: Union[None, bool] = None,
    ):
        return super().call(inputs, training=training)

    def reset_moving_stats(self):
        """
        Rescales the moving statistics of the BatchNormalization-Layer to zero mean and unit variance.
        """
        self.moving_mean.assign(tf.zeros(self.moving_mean.shape))
        self.moving_variance.assign(tf.ones(self.moving_variance.shape))
