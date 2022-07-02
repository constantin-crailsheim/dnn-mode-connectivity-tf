import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import mode_connectivity.curves.curves as curves
from mode_connectivity.argparser import (
    Arguments,
    parse_config,
    parse_evaluate_arguments,
    parse_train_arguments,
)
from mode_connectivity.curves.net import CurveNet
from mode_connectivity.data import data_loaders
from mode_connectivity.models.cnn import CNN
from mode_connectivity.models.mlp import MLP
from mode_connectivity.utils import (
    AlphaSchedule,
    load_checkpoint,
    save_checkpoint,
    save_weights,
    set_seeds,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train(args: Arguments):
    logger.info("Starting train")
    set_seeds(seed=args.seed)
    loaders, model = get_model_and_loaders(args)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=AlphaSchedule(args.lr, args.epochs), momentum=args.momentum
    )

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            checkpoint_path=args.resume, model=model, optimizer=optimizer
        )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.dir,
        save_weights_only=True,
        save_freq=args.save_freq,
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.fit(
        loaders["train"],
        validation_data=loaders["test"],
        epochs=args.epochs,
        initial_epoch=start_epoch,
        workers=args.num_workers,
        callbacks=[model_checkpoint_callback],
    )

    end_epoch = start_epoch + args.epochs
    # TODO is this double call necessary, or does checkpoint save weights as well?
    save_weights(directory=args.dir, epoch=end_epoch, model=model)
    save_checkpoint(
        directory=args.dir, epoch=end_epoch, model=model, optimizer=optimizer
    )


def train_cli():
    args = parse_train_arguments()
    train(args=args)


def train_from_config():
    args = parse_config()
    train(args=args)


def evaluate(args: Arguments):
    logger.info("Starting evaluate")
    set_seeds(seed=args.seed)
    loaders, model = get_model_and_loaders(args)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    # TODO Do we really want to evaluate on the train set?
    # r = model.evaluate_points(
    #     loaders["train"],
    #     num_points=args.num_points,
    #     point_on_curve=args.point_on_curve,
    #     verbose=True,
    #     return_dict=True,
    # )
    # print(r)
    r = model.evaluate_points(
        loaders["test"],
        num_points=args.num_points,
        point_on_curve=args.point_on_curve,
        verbose=True,
        return_dict=True,
    )
    print(r)


def get_model_and_loaders(args: Arguments):
    loaders, num_classes, _ = data_loaders(
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

    return loaders, model


def evaluate_cli():
    args = parse_evaluate_arguments()
    evaluate(args=args)


def evaluate_from_config():
    args = parse_config()
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
