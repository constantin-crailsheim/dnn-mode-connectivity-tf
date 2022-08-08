import os
import time
from typing import Callable, Dict, Iterable, Tuple, Union, List

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
    get_epoch,
    learning_rate_schedule,
    save_weights,
    set_seeds,
)

def main():
    """
    Initializes the variables necessary for the training procedure and triggers it.
    Customized for classification tasks.
    """
    args = parse_train_arguments()
    if args.disable_gpu:
        disable_gpu()
        
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
        input_shape=input_shape,
    )

    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=args.lr,
        momentum=args.momentum,
    )

    start_epoch = get_epoch(args)

    if not args.ckpt:
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
    """
    Carries out the training procedure for several epochs.

    Args:
        args (Arguments): Parser arguments.
        loaders (Dict[str, Iterable]): Data loaders.
        model (Layer): Model to be trained.
        criterion (Callable): Utilized loss function.
        optimizer (Optimizer): Optimizer.
        start_epoch (int): Initial value of epoch to start from when training.
        n_datasets (Dict): Amount of samples per data split.
    """
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
        if not args.curve and not has_batch_normalization:
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
    """
    Helper method for train().
    Carries out the training procedure for several mini-batches of an epoch and evaluates on the training set.

    Args:
        train_loader (Iterable): Data loader for the training set.
        model (Layer): Model to be trained.
        optimizer (Optimizer): Optimizer
        criterion (Callable): Utilized loss function.
        n_train (int): Amount of samples in the training set.
        lr_schedule (Union[Callable, None], optional): Learning rate schedule. Defaults to None.

    Returns:
        Dict[str, float]: Mean loss and accuracy of the epoch on the training set.
    """
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)

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
        "loss": loss_sum / n_train,
        "accuracy": correct * 100.0 / n_train,
    }


def test_epoch(
    test_loader: Iterable,
    model: Layer,
    criterion: Callable,
    n_test: int,
) -> Dict[str, float]:
    """
    Helper method for train().
    Evaluates the model on the test set after each training epoch in order to get unbiased estimates of the performance.
    This evaluation procedure is split into several mini-batches.

    Args:
        test_loader (Iterable):  Data loader for the test set.
        model (Layer): Model to be trained.
        criterion (Callable): Utilized loss function.
        n_test (int): Amount of samples in the test set.

    Returns:
        Dict[str, float]: Mean loss and accuracy of the epoch on the test set.
    """

    loss_sum = 0.0
    correct = 0.0

    for input, target in test_loader:
        loss_batch, correct_batch = test_batch(
            input=input,
            target=target,
            model=model,
            criterion=criterion,
        )
        loss_sum += loss_batch
        correct += correct_batch

    return {
        "loss": loss_sum / n_test,
        "accuracy": correct * 100.0 / n_test,
    }


def train_batch(
    input: tf.Tensor,
    target: tf.Tensor,
    model: Layer,
    optimizer: Optimizer,
    criterion: Callable,
) -> Tuple[float, float]:
    """
    Helper method for train_epoch().
    Performs Backpropagation and the SGD-Step for a mini-batch.

    Args:
        input (tf.Tensor): Input data that is propagated through the network leading to the network output.
        target (tf.Tensor): Targets which are compared to network output.
        model (Layer): Model to be trained.
        optimizer (Optimizer): Optimizer
        criterion (Callable): Utilized loss function.

    Returns:
        Tuple[float, float]: Batchwise metrics for the loss and accuracy on the training set. 
    """

    with tf.GradientTape() as tape:
        output = model(input, training=True)
        loss = criterion(target, output)
        loss += tf.add_n(model.losses)  # Add Regularization loss
    grads = tape.gradient(loss, model.trainable_variables)
    grads_and_vars = zip(grads, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars)

    loss = loss.numpy() * len(input)
    pred = tf.math.argmax(output, axis=1, output_type=tf.dtypes.int64)
    correct = tf.math.reduce_sum(
        tf.cast(tf.math.equal(pred, target), tf.float32)
    ).numpy()

    return loss, correct


def test_batch(
    input: tf.Tensor,
    target: tf.Tensor,
    model: Layer,
    criterion: Callable,
) -> Dict[str, float]:
    """
    Helper method for test_epoch().
    Batchwise computations for the loss and accuracy on the test set.

    Args:
        input (tf.Tensor): Test data that is propagated through the network leading to the network output.
        target (tf.Tensor): Test targets which are compared to network output.
        model (Layer): Model to be trained.
        criterion (Callable): Utilized loss function.

    Returns:
        Dict[str, float]: Batchwise metrics for the loss and accuracy on the test set. 
    """
    output = model(input, training=False) 
    loss = criterion(target, output)
    loss += tf.add_n(model.losses)  # Add Regularization loss

    loss = loss.numpy() * len(input)
    pred = tf.math.argmax(output, axis=1, output_type=tf.dtypes.int64)
    correct = tf.math.reduce_sum(
        tf.cast(tf.math.equal(pred, target), tf.float32)
    ).numpy()

    return loss, correct


def print_epoch_stats(values, epoch: int, start_epoch: int):
    """
    Helper method for train() that displays relevant statistics of an epoch.

    Args:
        values (List): Statistics to be displayed.
        epoch (int): Current epoch.
        start_epoch (int): Start epoch.
    """
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
