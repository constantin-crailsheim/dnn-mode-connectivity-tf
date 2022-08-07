import logging
import os
from functools import partial
from typing import Any, List, Union

import keras
import tensorflow as tf
from tensorflow.python.framework.errors import NotFoundError

import mode_connectivity.curves.curves as curves
from mode_connectivity.argparser import Arguments
from mode_connectivity.curves.net import CurveNet
from mode_connectivity.data import data_loaders
from mode_connectivity.models.cnn import CNN
from mode_connectivity.models.mlp import MLP

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def disable_gpu():
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
    # TODO torch.cuda.manual_seed(args.seed)


def learning_rate_schedule(base_lr, epoch, total_epochs):
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
        self.model = model
        self.total_epochs = total_epochs
        self.verbose = verbose
        self.base_lr = self.get_current_lr()
        self.schedule = partial(
            learning_rate_schedule, base_lr=self.base_lr, total_epochs=self.total_epochs
        )

    def get_current_lr(self) -> float:
        return float(tf.keras.backend.get_value(self.model.optimizer.lr))

    def on_epoch_begin(self, epoch: int, logs=None):
        lr = self.get_current_lr()
        # tf epoch is 0-indexed, so we need to add 1 to get the
        # same behaviour as in original implementation.
        new_lr = self.schedule(epoch=epoch + 1)

        if lr != new_lr:
            lr = new_lr
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        if self.verbose:
            print(f" lr: {lr:.4f}", end=" - ")


def l2_regularizer(weight_decay):
    return lambda model: 0.5 * weight_decay * model.l2


def adjust_learning_rate(optimizer, lr):
    optimizer.lr.assign(lr)
    return lr


def check_batch_normalization(model):
    return False


def load_checkpoint(
    checkpoint_path: str,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    **kwargs,
) -> int:
    """Load the model and optimizer from a saved checkpoint.
    The model and optimizer objects get updated with the stored parameters from the checkpoint.

    Also allows for additional parameters to load via kwargs.

    Args:
        checkpoint_path (str): Path to the checkpoint.
        model (keras.Model): The model which should be restored.
        optimizer (keras.optimizers.Optimizer): The optimizer which should be restored.

    Returns:
        int: The next epoch, from which training should be resumed.
    """
    logger.info(f"Restoring train state from {checkpoint_path}")
    epoch = tf.Variable(0, name="epoch")
    checkpoint = tf.train.Checkpoint(
        epoch=epoch, model=model, optimizer=optimizer, **kwargs
    )
    try:
        checkpoint.restore(checkpoint_path)
    except NotFoundError as e:
        logger.error(
            f"Could not restore specified checkpoint from {checkpoint_path}. Error: {e.message}"
        )
        logger.info("Starting from epoch 1")
        return 1

    next_epoch = int(epoch) + 1
    logger.info(f"Restored checkpoint, resuming from epoch {next_epoch}")
    return next_epoch


def save_checkpoint(
    directory: str,
    epoch: int,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    name: str = "checkpoint",
    **kwargs,
) -> None:
    """Save the current train state as a checkpoint.

    Also allows for additional parameters to be saved via kwargs.

    Args:
        directory (str): Directory where the checkpoint should be saved.
        epoch (int): The current train epoch.
        model (keras.Model): The trained model.
        optimizer (keras.optimizers.Optimizer): The optimizer used in training.
        name (str, optional): Custom name of the checkpoint. Defaults to "checkpoint".
    """
    checkpoint = tf.train.Checkpoint(
        epoch=tf.Variable(epoch, name="epoch"),
        model=model,
        optimizer=optimizer,
        **kwargs,
    )
    checkpoint_path = os.path.join(directory, f"{name}-epoch{epoch}")
    logger.info(f"Saving checkpoint to {checkpoint_path}")
    checkpoint.save(checkpoint_path)


def split_list(list_: List[Any], size: int) -> List[List[Any]]:
    """Split a list into equal chunks of size 'size'.

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


def save_model(
    directory: str,
    epoch: int,
    model: tf.keras.Model,
) -> None:
    """
    Save the current model in SavedModel format.
    Can only be called once the input dimension is specified.

    Args:
        directory (str): Directory where the checkpoint should be saved.
        epoch (int): The current train epoch.
        model (keras.Model): The trained model.
    """
    model_path = os.path.join(directory, f"model-epoch{epoch}")
    logger.info(f"Saving model to {model_path}")
    # print(model.curve_model.input_spec)
    # print(model.call.get_concrete_function(inputs=model.curve_model.input_spec))
    model.save(model_path)


def save_weights(
    directory: str,
    epoch: int,
    model: keras.Model,
):
    model_path = os.path.join(directory, f"model-weights-epoch{epoch}")
    logger.info(f"Saving model weights to {model_path}")
    # print(model.curve_model.input_spec)
    # print(model.call.get_concrete_function(inputs=model.curve_model.input_spec))
    model.save_weights(model_path)


def get_architecture(model_name: str):
    if model_name == "CNN":
        return CNN
    if model_name == "MLP":
        return MLP
    raise KeyError(f"Unkown model {model_name}")


def get_model(architecture, args: Arguments, num_classes: Union[int, None], input_shape):
    # If no curve is to be fit the base version of the architecture is initialised (e.g CNNBase instead of CNNCurve).
    if not args.curve:
        logger.info(f"Loading Regular Model {architecture.__name__}")
        model = architecture.base(
            num_classes=num_classes, weight_decay=args.wd, **architecture.kwargs
        )
        return model

    # Otherwise the curve version of the architecture (e.g. CNNCurve) is initialised in the context of a CurveNet.
    # The CurveNet additionally contains the curve (e.g. Bezier) and imports the parameters of the pre-trained base-nets that constitute the outer points of the curve.
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
        logger.info(f"Restoring model from checkpoint {args.ckpt} for evaluation")
        # Evalutate the model from Checkpoint
        # expect_partial()
        # -> silence Value in checkpoint could not be found in the restored object: (root).optimizer. ..
        # https://stackoverflow.com/questions/58289342/tf2-0-translation-model-error-when-restoring-the-saved-model-unresolved-objec
        model.load_weights(args.ckpt).expect_partial()
        return model

    # Build model from 0, 1 or 2 base_models
    base_model = architecture.base(
        num_classes=num_classes, weight_decay=args.wd, **architecture.kwargs
    )
    base_model.build(input_shape=input_shape)
    if args.resume is None:
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

    return model


def load_base_weights(
    path: str, index: int, model: tf.keras.Model, base_model: tf.keras.Model
) -> None:
    if not path:
        return None
    logger.info(f"Loading {path} as point #{index}")
    base_model.load_weights(path).expect_partial()
    model.import_base_parameters(base_model, index)


def get_model_and_loaders(args: Arguments):
    loaders, num_classes, _, input_shape = data_loaders(
        dataset=args.dataset,
        path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform_name=args.transform,
        use_test=args.use_test,
    )

    architecture = get_architecture(model_name=args.model)
    model = get_model(
        architecture=architecture,
        args=args,
        num_classes=num_classes,
        input_shape=input_shape,
    )

    return loaders, model
