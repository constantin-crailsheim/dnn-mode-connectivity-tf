import os
from typing import Callable, Dict, Iterable

import numpy as np
import tabulate

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from showcase.argparser import parse_evaluate_arguments
from showcase.data import data_loaders
from showcase.utils import disable_gpu, get_architecture, get_model, set_seeds


def main():
    """
    Initializes the variables necessary for the evaluation procedure and triggers it.
    Customized for regression tasks.
    """
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
            train_losses,
            test_losses,
            points_on_curve,
            args.dir,
            args.file_name_appendix,
        )


def evaluate_epoch(
    test_loader: Iterable, model: tf.keras.Model, criterion: Callable, n_test: int
) -> Dict[str, tf.Tensor]:
    """Evaluation of epoch for loaded model.

    Args:
        test_loader (Iterable): Data loaders with minibatches.
        model (Layer): Model to be evaluated.
        criterion (Callable): Utilized loss function.
        n_test (int): Amount of example in dataset evaluated.

    Returns:
        Dict[str]: Evaluated loss.
    """
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
    """
    Helper method for evaluate_epoch().
    Batchwise computations for the loss, predictions, output and target on the dataset evaluated.

    Args:
        input (tf.Tensor): Data that is propagated through the network leading to the network output.
        target (tf.Tensor): Targets which are compared to network output.
        model (Layer): Model to be trained.
        criterion (Callable): Utilized loss function.

    Returns:
        float: Evaluated loss.
    """

    output = model(input, training=False)
    loss = criterion(target, output)

    loss = loss.numpy() * len(input)

    return loss


def print_stats_of_point_on_curve(values, i):
    """
    Displays relevant statistics of an epoch.

    Args:
        values (List): Statistics to be displayed.
        epoch (int): Current epoch.
    """
    columns = ["Point on curve", "Train loss", "Test loss"]
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="10.4f")
    if i % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)


def save_stats_of_points_on_curve(
    train_losses, test_losses, points_on_curve, dir: str, file_name_appendix: str
):
    """
    Save relevants statistics of point on curve.

    Args:
        train_losses (numpy.ndarray): Array of train losses.
        test_losses (numpy.ndarray): Array of test losses.
        points_on_curve (numpy.ndarray): Array of points on curve evaluated.
        dir (str): Directory to store file.
        file_name (str, optional): Name of file. Defaults to 'stats_of_points_on_curve.npz'.
    """
    file_name = "stats_of_points_on_curve" + file_name_appendix + ".npz"
    os.makedirs(dir, exist_ok=True)
    np.savez(
        os.path.join(dir, file_name),
        points_on_curve=points_on_curve,
        train_losses=train_losses,
        test_losses=test_losses,
    )


if __name__ == "__main__":
    main()
