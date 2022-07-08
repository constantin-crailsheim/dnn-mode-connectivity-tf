import os
import time
from typing import Callable, Dict, Iterable, Tuple, Union

import tabulate
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from keras.layers import Layer
from keras.optimizers import Optimizer
from mode_connectivity.argparser import Arguments, parse_train_arguments
from mode_connectivity.data import data_loaders
from mode_connectivity.utils import (
    adjust_learning_rate,
    check_batch_normalization,
    disable_gpu,
    get_architecture,
    get_model,
    learning_rate_schedule,
    load_checkpoint,
    save_weights,
    set_seeds,
)


def main():
    args = parse_train_arguments()
    if args.disable_gpu:
        disable_gpu()

    # TODO: Set backends cudnnn
    set_seeds(seed=args.seed)

    loaders, num_classes, n_datasets, input_shape = data_loaders(
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
        input_shape=input_shape,  # TODO Determine this from dataset
    )

    # criterion = tf.nn.sparse_softmax_cross_entropy_with_logits
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # TODO: Check if correct function, takes labels of shape [nbatch, nclass], while F.cross_entropy()
    # takes labels of shape [nBatch]
    optimizer = tf.keras.optimizers.SGD(
        # TODO how can we fit equivalent of arg params in PyTorch
        # PyTorch: params=filter(lambda param: param.requires_grad, model.parameters()),
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        learning_rate=args.lr,
        momentum=args.momentum,
        # Weight decay added into models/mlp.py
        # https://stackoverflow.com/questions/55046234/sgd-with-weight-decay-parameter-in-tensorflow
        # https://d2l.ai/chapter_multilayer-perceptrons/weight-decay.html
        # PyTorch: weight_decay=args.wd if args.curve is None else 0.0,
    )
    # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD

    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(
            checkpoint_path=args.resume, model=model, optimizer=optimizer
        )
    save_weights(directory=args.dir, epoch=start_epoch - 1, model=model)

    train(
        args=args,
        loaders=loaders,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        start_epoch=start_epoch,
        n_datasets=n_datasets,
    )


def train(
    args: Arguments,
    loaders: Dict[str, Iterable],
    model: Layer,
    criterion: Callable,
    optimizer: Optimizer,
    start_epoch: int,
    n_datasets: Dict,
):
    has_batch_normalization = check_batch_normalization(
        model
    )  # Not implemented yet, returns always False
    test_results = {"loss": None, "accuracy": None, "nll": None}

    for epoch in range(start_epoch, args.epochs + 1):
        time_epoch = time.time()

        lr = learning_rate_schedule(args.lr, epoch, args.epochs)
        adjust_learning_rate(optimizer, lr)

        train_results = train_epoch(
            loaders["train"],
            model,
            optimizer,
            criterion,
            n_datasets["train"],
        )

        # Does the condition make sense here?
        if not args.curve or not has_batch_normalization:
            test_results = test_epoch(
                loaders["test"], model, criterion, n_datasets["test"]
            )

        if epoch % args.save_freq == 0:
            save_weights(directory=args.dir, epoch=epoch, model=model)

        time_epoch = time.time() - time_epoch
        values = [
            epoch,
            lr,
            train_results["loss"],
            train_results["accuracy"],
            test_results["loss"],
            test_results["accuracy"],
            time_epoch,
        ]

        print_epoch_stats(values, epoch, start_epoch)

    if args.epochs % args.save_freq != 0:
        # Save last checkpoint if not already saved
        save_weights(directory=args.dir, epoch=epoch, model=model)

def train_epoch(
    train_loader: Iterable,
    model: Layer,
    optimizer: Optimizer,
    criterion: Callable,
    n_train: int,
    lr_schedule: Union[Callable, None] = None, # Do we need this?
) -> Dict[str, float]:
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    # PyTorch: model.train()

    for iter, (input, target) in enumerate(train_loader):
        if callable(lr_schedule):
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        loss_batch, correct_batch = train_batch(
            input=input,
            target=target,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
        )
        loss_sum += loss_batch
        correct += correct_batch

    return {
        "loss": loss_sum / n_train,  # Add function to find length
        "accuracy": correct * 100.0 / n_train,  # Add function to find length of dataset,
    }


def test_epoch(
    test_loader: Iterable,
    model: Layer,
    criterion: Callable,
    n_test: int,
    **kwargs,
) -> Dict[str, float]:
    loss_sum = 0.0
    correct = 0.0

    # PyTorch: model.eval()
    for input, target in test_loader:
        loss_batch, correct_batch = test_batch(
            input=input,
            target=target,
            model=model,
            criterion=criterion,
            **kwargs,
        )
        loss_sum += loss_batch
        correct += correct_batch

    return {
        "loss": loss_sum / n_test,  # Add function to find length
        "accuracy": correct * 100.0 / n_test,  # Add function to find length
    }


def train_batch(
    input: tf.Tensor,
    target: tf.Tensor,
    model: Layer,
    optimizer: Optimizer,
    criterion: Callable,
) -> Tuple[float, float]:
    with tf.GradientTape() as tape:
        output = model(input)
        loss = criterion(target, output)
        loss += tf.add_n(model.losses)  # Add Regularization loss
    grads = tape.gradient(loss, model.trainable_variables)
    grads_and_vars = zip(grads, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars)

    # See for above:
    # https://medium.com/analytics-vidhya/3-different-ways-to-perform-gradient-descent-in-tensorflow-2-0-and-ms-excel-ffc3791a160a
    # https://d2l.ai/chapter_multilayer-perceptrons/weight-decay.html (4.5.4)

    loss = loss.numpy() * len(input)
    pred = tf.math.argmax(output, axis=1, output_type=tf.dtypes.int64)
    # Is there an easier way?
    correct = tf.math.reduce_sum(
        tf.cast(tf.math.equal(pred, target), tf.float32)
    ).numpy()

    return loss, correct  # Do we need to return the model as well?


def test_batch(
    input: tf.Tensor,
    target: tf.Tensor,
    model: Layer,
    criterion: Callable,
    **kwargs,
) -> Dict[str, float]:
    output = model(input, **kwargs)
    # TODO is Negative Loss Likelihood calculated correctly here?
    loss = criterion(target, output)
    loss += tf.add_n(model.losses)  # Add Regularization loss

    loss = loss.numpy() * len(input)
    pred = tf.math.argmax(output, axis=1, output_type=tf.dtypes.int64)
    correct = tf.math.reduce_sum(
        tf.cast(tf.math.equal(pred, target), tf.float32)
    ).numpy()

    return loss, correct


def print_epoch_stats(values, epoch, start_epoch):
    COLUMNS = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time"]
    table = tabulate.tabulate([values], COLUMNS, tablefmt="simple", floatfmt="9.4f")
    if epoch % 40 == 1 or epoch == start_epoch:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)


if __name__ == "__main__":
    main()
