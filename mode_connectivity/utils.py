import logging
import os

import keras
import tensorflow as tf
from tensorflow.python.framework.errors import NotFoundError

logger = logging.getLogger(__name__)


def learning_rate_schedule(base_lr, epoch, total_epochs):
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr


def l2_regularizer(weight_decay):
    return lambda model: 0.5 * weight_decay * model.l2


def adjust_learning_rate(optimizer, lr):
    optimizer.lr.assign(lr)
    return lr


def check_batch_normalization(model):
    return True


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
