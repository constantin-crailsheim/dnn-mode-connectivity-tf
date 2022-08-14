import logging
import os
import random
from functools import partial
from typing import Any, List, Union

import keras
import mode_connectivity.curves as curves
import tensorflow as tf
from mode_connectivity.net import CurveNet

from showcase.argparser import Arguments
from showcase.data import data_loaders
from showcase.models.cnn import CNN
from showcase.models.cnnbn import CNNBN
from showcase.models.mlp import MLP

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def disable_gpu():
    """GPU is used as default for tensorflow. Disables GPU and uses CPU instead."""
    logger.info("Trying to disable GPU")
    try:
        tf.config.set_visible_devices([], "GPU")
        logger.info(f"GPU disabled.")
    except RuntimeError:
        logger.error(
            "Cannot modify devices after calling tensorflow methods. "
            "Try to set your device before any operations. "
        )
    logger.info(f"Running on devices {tf.config.get_visible_devices()}")


def set_seeds(seed: int):
    tf.random.set_seed(seed)

def learning_rate_schedule(base_lr: float, epoch: int, total_epochs: int):
    """
    Determines the learning rate for an epoch  based on an initial learning rate and total amount of epochs.
    The implemented schedule monotonically decreases the learning rate to a factor of 0.01.

    Args:
        base_lr (float): The initial learning rate.
        epoch (int): Current epoch.
        total_epochs (int): Total amount of epochs.

    Returns:
        float: Scheduled learning rate for the current epoch.
    """
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr


class AlphaLearningRateSchedule(tf.keras.callbacks.Callback):
    def __init__(self, model: tf.keras.Model, total_epochs: int, verbose: bool = True):
        """
        Initializes the AlphaLearningRateSchedule.

        Args:
            model (tf.keras.Model): Currently trained model.
            total_epochs (int): Total amount of epochs.
            verbose (bool, optional): Indicates if updates are displayed. Defaults to True.
        """
        self.model = model
        self.total_epochs = total_epochs
        self.verbose = verbose
        self.base_lr = self.get_current_lr()
        self.schedule = partial(
            learning_rate_schedule, base_lr=self.base_lr, total_epochs=self.total_epochs
        )

    def get_current_lr(self) -> float:
        """
        Returns the current learning rate of the trained model.

        Returns:
            float: Current learning rate.
        """
        return float(tf.keras.backend.get_value(self.model.optimizer.lr))

    def on_epoch_begin(self, epoch: int, logs=None):
        """
        Triggers the learning rate update at the beginning of each epoch.

        Args:
            epoch (int): Current epoch.
            logs (_type_, optional): Not used.
        """
        lr = self.get_current_lr()
        # tf epoch is 0-indexed, so we need to add 1 to get the
        # same behaviour as in original implementation.
        new_lr = self.schedule(epoch=epoch + 1)

        if lr != new_lr:
            lr = new_lr
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        if self.verbose:
            print(f" lr: {lr:.4f}", end=" - ")


class PointOnCurveMetric(tf.keras.metrics.Metric):
    """Custom Metric to display the current point on curve (poc) used in training/evaluation."""

    def __init__(self, model: tf.keras.Model, **kwargs):
        super().__init__(name="poc", **kwargs)
        self.model = model

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def result(self):
        return self.model.point_on_curve


def adjust_learning_rate(optimizer, lr):
    """
    Adjust the learning rate used in an optimizer.

    Args:
        optimizer (Optimizer): Optimizer to be updated.
        lr (float): Current learning rate as computed by a learning rate schedule.

    Returns:
        float: The adjusted learning rate.
    """
    optimizer.lr.assign(lr)


def split_list(list_: List[Any], size: int) -> List[List[Any]]:
    """Split a list into equal chunks of specified size.

    Args:
        list_ (List[Any]): The list to split.
        size (int): The chunk size.

    Returns:
        List[List[Any]]: List with equal sized chunks.

    Example:
    ```python
    split_list([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    ```
    """
    return list(list_[i : i + size] for i in range(0, len(list_), size))


def save_weights(
    directory: str,
    epoch: int,
    model: keras.Model,
):
    """
    Saves the model's weights/parameters to a specified path in order to be restored for curve fitting or evaluation.

    Args:
        directory (str): Directory to store the weights/parameters.
        epoch (int): Current epoch of training.
        model (keras.Model): Current model to extract parameters from.
    """
    model_path = os.path.join(directory, f"model-weights-epoch{epoch}")
    logger.info(f"Saving model weights to {model_path}")
    model.save_weights(model_path)


def get_architecture(model_name: str):
    """
    For a specified model name returns the corresponding architecture that is used to fit the model.
    The architecture consists of the Base and Curve version of the model.

    Args:
        model_name (str): Name of the model. One out of {"CNN", "CNNBN", "MLP"}.

    Raises:
        KeyError: Indicates if the model name is unknown/ the model is not implemented yet.

    Returns:
        _type_: Model architecture.
    """
    if model_name == "CNN":
        return CNN
    if model_name == "CNNBN":
        return CNNBN
    if model_name == "MLP":
        return MLP
    raise KeyError(f"Unkown model {model_name}")


def get_model(
    architecture, args: Arguments, num_classes: Union[int, None], input_shape
):
    """
    Initializes and returns either the Base or Curve version of an architecture.

    Args:
        architecture (_type_): Model architecture, e.g. CNN.
        args (Arguments): Parsed arguments.
        num_classes (Union[int, None]): The amount of classes the net discriminates among.  Specified as "None" in regression tasks.
        input_shape (tf.Tensor): Shape of the input data.

    Returns:
        _type_: Initialized model.
    """
    # If no curve is to be fit the base version of the architecture is initialized (e.g CNNBase instead of CNNCurve).
    if args.resume_epoch and not args.ckpt:
        raise KeyError("Checkpoint needs to be defined to resume training.")

    if not args.curve:
        logger.info(f"Loading regular model {architecture.__name__}")
        model = architecture.base(
            num_classes=num_classes, weight_decay=args.wd, **architecture.kwargs
        )
        model.compile()
        if args.ckpt:
            logger.info(f"Restoring regular model from checkpoint {args.ckpt}.")
            model.build(input_shape=input_shape)
            model.load_weights(filepath=args.ckpt).expect_partial()
            model.compile()
        return model

    # Otherwise the curve version of the architecture (e.g. CNNCurve) is initialized in the context of a CurveNet.
    # The CurveNet additionally contains the curve (e.g. Bezier) and imports parameters from the pre-trained base-nets that constitute the outer points of the curve.
    curve = getattr(curves, args.curve)
    logger.info(
        f"Loading CurveNet with CurveModel {architecture.__name__} and Curve {curve.__name__}"
    )
    model = CurveNet(
        num_classes=num_classes,
        num_bends=args.num_bends,
        weight_decay=args.wd,
        curve=curve,
        curve_model=architecture.curve,
        fix_start=args.fix_start,
        fix_end=args.fix_end,
        architecture_kwargs=architecture.kwargs,
    )

    if args.ckpt:
        logger.info(f"Restoring curve model from checkpoint {args.ckpt}.")
        model.build(input_shape=input_shape)
        model.load_weights(filepath=args.ckpt).expect_partial()
        model.compile()
    else:
        # Build model from 0, 1 or 2 base_models
        base_model = architecture.base(
            num_classes=num_classes, weight_decay=args.wd, **architecture.kwargs
        )
        base_model.build(input_shape=input_shape)
        load_base_weights(
            path=args.init_start,
            index=0,
            model=model,
            base_model=base_model,
        )
        load_base_weights(
            path=args.init_end,
            index=args.num_bends + 1,
            model=model,
            base_model=base_model,
        )
        if args.init_linear:
            logger.info("Linear initialization.")
            model.init_linear()
            model.compile()

    return model


def get_epoch(args: Arguments):
    if args.ckpt:
        if args.resume_epoch:
            return args.resume_epoch
        else:
            raise KeyError("Start epoch to resume training needs to be specified")
    else:
        return 1


def load_base_weights(
    path: str, index: int, model: tf.keras.Model, base_model: tf.keras.Model
) -> None:
    """
    Helper method for get_model().
    Loads the weights/parameters of a BaseModel and assigns them to a curve node.

    Args:
        path (str): Path to load weights/ parameters from.
        index (int): Index indicating the bend/ point of curve.
        model (tf.keras.Model): Model to be assigned with weights/parameters.
        base_model (tf.keras.Model): Model to load weights/parameters from.
    """
    if not path:
        return None
    logger.info(f"Loading {path} as point #{index}")
    base_model.load_weights(path).expect_partial()
    model.import_base_parameters(base_model, index)


def get_model_and_loaders(args: Arguments):
    """
    Returns data loaders and a model based on parser arguments.

    Args:
        args (Arguments): Parser arguments.

    Returns:
        _type_: Tuple of data loaders and model.
    """
    loaders, num_classes, _, input_shape = data_loaders(
        dataset=args.dataset,
        path=args.data_path,
        batch_size=args.batch_size,
    )

    architecture = get_architecture(model_name=args.model)
    model = get_model(
        architecture=architecture,
        args=args,
        num_classes=num_classes,
        input_shape=input_shape,
    )

    return loaders, model
