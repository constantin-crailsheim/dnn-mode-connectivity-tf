import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from mode_connectivity.argparser import Arguments, parse_train_arguments
from mode_connectivity.logger import configure_loggers
from mode_connectivity.utils import (
    AlphaLearningRateSchedule,
    PointOnCurveMetric,
    disable_gpu,
    get_epoch,
    get_model_and_loaders,
    save_weights,
    set_seeds,
)

logger = logging.getLogger(__name__)


def train(args: Arguments):
    configure_loggers()
    if args.disable_gpu:
        disable_gpu()

    logger.info("Starting train")
    set_seeds(seed=args.seed)

    loaders, model = get_model_and_loaders(args)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=args.momentum)

    start_epoch = (
        get_epoch(args) - 1
    )  # tf epoch is 0-indexed, load_checkpoint's min return is 1

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.dir,
        save_weights_only=True,
        save_freq=args.save_freq,
    )

    model.compile(
        optimizer=optimizer, loss=loss, metrics=["accuracy", PointOnCurveMetric(model)]
    )
    learning_rate_scheduler = AlphaLearningRateSchedule(model, total_epochs=args.epochs)
    model.fit(
        loaders["train"],
        validation_data=loaders["test"],
        epochs=args.epochs,
        initial_epoch=start_epoch,
        callbacks=[model_checkpoint_callback, learning_rate_scheduler],
    )

    save_weights(directory=args.dir, epoch=args.epochs, model=model)


if __name__ == "__main__":
    args = parse_train_arguments()
    train(args=args)
