import logging

import tensorflow as tf

import mode_connectivity.curves.curves as curves
from mode_connectivity.argparser import (
    Arguments,
    parse_evaluate_arguments,
    parse_train_arguments,
)
from mode_connectivity.curves.net import CurveNet
from mode_connectivity.data import data_loaders
from mode_connectivity.models.cnn import CNN
from mode_connectivity.models.mlp import MLP
from mode_connectivity.utils import load_checkpoint, save_model, save_weights, set_seeds

logger = logging.getLogger(__name__)


def train(args: Arguments):
    set_seeds(seed=args.seed)
    loaders, num_classes, n_datasets = data_loaders(
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
        input_shape=(None, 28, 28, 1),  # TODO Determine this from dataset
    )

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=args.momentum)

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            checkpoint_path=args.resume, model=model, optimizer=optimizer
        )

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    model.fit(
        loaders["train"],
        validation_data=loaders["test"],
        epochs=args.epochs,
        initial_epoch=start_epoch,
        workers=args.num_workers,
    )

    # TODO Add save checkpoints

    save_weights(directory=args.dir, epoch=start_epoch + args.epochs, model=model)
    save_model(directory=args.dir, epoch=start_epoch + args.epochs, model=model)


def train_cli():
    args = parse_train_arguments()
    train(args=args)


def evaluate(args: Arguments):
    set_seeds(seed=args.seed)
    loaders, num_classes, n_datasets = data_loaders(
        dataset=args.dataset,
        path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform_name=args.transform,
        use_test=args.use_test,
    )


def evaluate_cli():
    args = parse_evaluate_arguments()
    evaluate(args=args)


def get_architecture(model_name: str):
    if model_name == "CNN":
        return CNN
    if model_name == "MLP":
        return MLP
    raise KeyError(f"Unkown model {model_name}")


def get_model(architecture, args: Arguments, num_classes: int, input_shape):
    # If no curve is to be fit the base version of the architecture is initialised (e.g CNNBase instead of CNNCurve).
    if not args.curve:
        model = architecture.base(
            num_classes=num_classes, weight_decay=args.wd, **architecture.kwargs
        )
        return model

    # Otherwise the curve version of the architecture (e.g. CNNCurve) is initialised in the context of a CurveNet.
    # The CurveNet additionally contains the curve (e.g. Bezier) and imports the parameters of the pre-trained base-nets that constitute the outer points of the curve.
    curve = getattr(curves, args.curve)
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
            index=args.num_bends + 2,
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
    base_model.load_weights(path)
    model.import_base_parameters(base_model, index)
