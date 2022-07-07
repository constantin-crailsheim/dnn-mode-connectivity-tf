import logging
import os
from typing import Dict, List

import tabulate

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tabulate
import tensorflow as tf

from mode_connectivity.argparser import (
    Arguments,
    parse_config,
    parse_evaluate_arguments,
    parse_train_arguments,
)
from mode_connectivity.curves.net import CurveNet
from mode_connectivity.data import data_loaders
from mode_connectivity.logger import configure_loggers
from mode_connectivity.models.cnn import CNN
from mode_connectivity.models.mlp import MLP
from mode_connectivity.utils import (
    AlphaLearningRateSchedule,
    disable_gpu,
    get_model_and_loaders,
    load_checkpoint,
    save_checkpoint,
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

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            checkpoint_path=args.resume, model=model, optimizer=optimizer
        )
        start_epoch -= 1  # tf epoch is 0-indexed, load_checkpoint's min return is 1

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.dir,
        save_weights_only=True,
        save_freq=args.save_freq,
    )

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    learning_rate_scheduler = AlphaLearningRateSchedule(model, total_epochs=args.epochs)
    model.fit(
        loaders["train"],
        validation_data=loaders["test"],
        epochs=args.epochs,
        initial_epoch=start_epoch,
        callbacks=[model_checkpoint_callback, learning_rate_scheduler],
    )

    # TODO is this double call necessary, or does checkpoint save weights as well?
    save_weights(directory=args.dir, epoch=args.epochs, model=model)
    save_checkpoint(
        directory=args.dir, epoch=args.epochs, model=model, optimizer=optimizer
    )


def train_cli():
    args = parse_train_arguments()
    train(args=args)


def train_from_config():
    args = parse_config()
    train(args=args)


def evaluate(args: Arguments):
    configure_loggers()
    if args.disable_gpu:
        disable_gpu()

    logger.info("Starting evaluate")
    set_seeds(seed=args.seed)

    loaders, model = get_model_and_loaders(args)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(loss=loss, metrics=["accuracy"])
    # TODO Do we really want to evaluate on the train set?
    results = model.evaluate_points(
        loaders["test"],
        num_points=args.num_points,
        point_on_curve=args.point_on_curve,
        verbose=True,
        return_dict=True,
    )
    print_evaluation_results(results)


def print_evaluation_results(results: List[Dict[str, float]]):
    headers = ["Point on curve", "Loss", "Accuracy"]
    metrics = ["point_on_curve", "loss", "accuracy"]
    d = [list(f"{r[m]:.4f}" for m in metrics) for r in results]

    table = tabulate.tabulate(
        d,
        headers=headers,
        tablefmt="pretty",
    )
    print(table)


def evaluate_cli():
    args = parse_evaluate_arguments()
    evaluate(args=args)


def evaluate_from_config():
    args = parse_config()
    evaluate(args=args)
