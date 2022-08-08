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
    get_epoch,
    learning_rate_schedule,
    save_weights,
    set_seeds,
)

def main():
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
        input_shape=input_shape
    )

    criterion = tf.keras.losses.MeanSquaredError()

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
    has_batch_normalization = check_batch_normalization(
        model
    )  # Not implemented yet, returns always False
    test_results = {"loss": None}

    for epoch in range(start_epoch, args.epochs + 1):
        time_epoch = time.time()

        lr = learning_rate_schedule(args.lr, epoch, args.epochs)
        adjust_learning_rate(optimizer, lr)

        train_results = train_epoch(
            loaders["train"],
            model,
            optimizer,
            criterion,
            n_datasets["train"]
        )

        if not args.curve or not has_batch_normalization:
            test_results = test_epoch(
                loaders["test"],
                model,
                criterion,
                n_datasets["test"]
            )

        if epoch % args.save_freq == 0:
            save_weights(directory=args.dir, epoch=epoch, model=model)
            
        time_epoch = time.time() - time_epoch
        values = [
            epoch,
            lr,
            train_results["loss"],
            test_results["loss"],
            time_epoch,
        ]

        print_epoch_stats(values, epoch, start_epoch)

    if args.epochs % args.save_freq != 0:
        # Save last weights if not already saved
        save_weights(directory=args.dir, epoch=epoch, model=model)

def train_epoch(
    train_loader: Iterable,
    model: Layer,
    optimizer: Optimizer,
    criterion: Callable,
    n_train: int,
    lr_schedule: Union[Callable, None] = None
) -> Dict[str, float]:
    loss_sum = 0.0

    num_iters = len(train_loader)

    for iter, (input, target) in enumerate(train_loader):
        if callable(lr_schedule):
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        
        loss_batch = train_batch(
            input=input,
            target=target,
            model=model,
            optimizer=optimizer,
            criterion=criterion
        )
        
        loss_sum += loss_batch

    return {
        "loss": loss_sum / n_train
    }


def test_epoch(
    test_loader: Iterable,
    model: Layer,
    criterion: Callable,
    n_test: int
) -> Dict[str, float]:
    loss_sum = 0.0

    for input, target in test_loader:
        loss_batch = test_batch(
            input=input,
            target=target,
            model=model,
            criterion=criterion
        )
        loss_sum += loss_batch

    return {
        "loss": loss_sum / n_test
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
        loss += tf.add_n(model.losses)

    grads = tape.gradient(loss, model.trainable_variables)
    grads_and_vars = zip(grads, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars)

    loss = loss.numpy() * len(input)

    return loss


def test_batch(
    input: tf.Tensor,
    target: tf.Tensor,
    model: Layer,
    criterion: Callable,
) -> Dict[str, float]:

    output = model(input)
    loss = criterion(target, output)
    loss += tf.add_n(model.losses)

    loss = loss.numpy() * len(input)
        
    return loss


def print_epoch_stats(values, epoch, start_epoch):
    COLUMNS = ["ep", "lr", "tr_loss", "te_loss", "time"]
    table = tabulate.tabulate([values], COLUMNS, tablefmt="simple", floatfmt="9.4f")
    if epoch % 40 == 1 or epoch == start_epoch:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)


if __name__ == "__main__":
    main()
