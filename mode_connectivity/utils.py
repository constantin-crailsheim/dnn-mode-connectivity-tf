import logging
import os
from functools import partial
from typing import Any, List

import keras
import tensorflow as tf
from tensorflow.python.framework.errors import NotFoundError

logger = logging.getLogger(__name__)


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
    model: keras.Model,
    optimizer: keras.optimizers.Optimizer,
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
    model: keras.Model,
    optimizer: keras.optimizers.Optimizer,
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
    model: keras.Model,
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
