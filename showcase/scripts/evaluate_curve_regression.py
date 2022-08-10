import os
from typing import Callable, Dict, Iterable, Union

import numpy as np
import tabulate

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from showcase.argparser import Arguments, parse_evaluate_arguments
from showcase.data import data_loaders
from showcase.utils import (
    disable_gpu,
    get_model,
    get_architecture
)


def main():
    args = parse_evaluate_arguments()
    if args.disable_gpu:
        disable_gpu()

    loaders, num_classes, n_datasets, input_shape = data_loaders(
        dataset=args.dataset, path=args.data_path, batch_size=args.batch_size
    )
    architecture = get_architecture(model_name=args.model)
    model = get_model(
        architecture=architecture,
        args=args,
        num_classes=num_classes,
        input_shape=input_shape,
    )

    criterion = tf.keras.losses.MeanSquaredError()

    if args.num_points != None:
        T = args.num_points
        points_on_curve = np.linspace(0.0, 1.0, T)
    elif args.point_on_curve != None:
        T = 1
        points_on_curve = [args.point_on_curve]

    train_losses = np.zeros(T)
    test_losses = np.zeros(T)

    for i, point_on_curve in enumerate(points_on_curve):
        point_on_curve_tensor = tf.constant(point_on_curve, shape=(), dtype=tf.float32)
        model.point_on_curve.assign(point_on_curve_tensor)

        train_results = evaluate_epoch(
            test_loader=loaders["train"],
            model=model,
            criterion=criterion,
            n_test=n_datasets["train"],
        )
        test_results = evaluate_epoch(
            test_loader=loaders["test"],
            model=model,
            criterion=criterion,
            n_test=n_datasets["test"],
        )

        train_losses[i] = train_results["loss"]
        test_losses[i] = test_results["loss"]

        values = [point_on_curve, train_losses[i], test_losses[i]]

        print_stats_of_point_on_curve(values, i)

    if args.save_evaluation == True:
        save_stats_of_points_on_curve(
            train_losses, test_losses, points_on_curve, args.dir
        )

def evaluate_epoch(
    test_loader: Iterable, model: tf.keras.Model, criterion: Callable, n_test: int
) -> Dict[str, tf.Tensor]:

    loss_sum = 0.0

    for input, target in test_loader:
        loss_batch = evaluate_batch(
            input=input, target=target, model=model, criterion=criterion
        )

        loss_sum += loss_batch

    return {"loss": loss_sum / n_test}


def evaluate_batch(
    input: tf.Tensor, target: tf.Tensor, model: tf.keras.Model, criterion: Callable
) -> Dict[str, float]:

    output = model(input, training=False)
    loss = criterion(target, output)

    loss = loss.numpy() * len(input)

    return loss


def print_stats_of_point_on_curve(values, i):
    columns = ["Point on curve", "Train loss", "Test loss"]
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="10.4f")
    if i % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)


def save_stats_of_points_on_curve(
    train_losses,
    test_losses,
    points_on_curve,
    dir: str,
    file_name: str = "stats_of_points_on_curve.npz",
):

    os.makedirs(dir, exist_ok=True)
    np.savez(
        os.path.join(dir, file_name),
        points_on_curve=points_on_curve,
        train_losses=train_losses,
        test_losses=test_losses,
    )


if __name__ == "__main__":
    main()
