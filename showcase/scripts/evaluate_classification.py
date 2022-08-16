import os
from typing import Callable, Dict, Iterable, Union

import numpy as np
import tabulate

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from showcase.argparser import Arguments, parse_evaluate_arguments
from showcase.data import data_loaders
from showcase.utils import disable_gpu, set_seeds, get_architecture, get_model


def main():
    """
    Initializes the variables necessary for the evaluation procedure and triggers it.
    Customized for classification tasks.
    """
    args = parse_evaluate_arguments()
    if args.disable_gpu:
        disable_gpu()
    set_seeds(args.seed)

    loaders, num_classes, n_datasets, input_shape = data_loaders(
        dataset=args.dataset,
        path=args.data_path,
        batch_size=args.batch_size,
    )
    architecture = get_architecture(model_name=args.model)
    model = get_model(
        architecture=architecture,
        args=args,
        num_classes=num_classes,
        input_shape=input_shape,
    )

    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if args.num_points != None:
        T = args.num_points
        points_on_curve = np.linspace(0.0, 1.0, T)
    elif args.point_on_curve != None:
        T = 1
        points_on_curve = [args.point_on_curve]

    train_losses = np.zeros(T)
    train_accuracy_scores = np.zeros(T)
    train_f1_scores = np.zeros(T)
    test_losses = np.zeros(T)
    test_accuracy_scores = np.zeros(T)
    test_f1_scores = np.zeros(T)

    dl = np.zeros(T)

    train_predictions = []
    train_targets = []
    train_output = []
    test_predictions = []
    test_targets = []
    test_output = []

    previous_parameters = None

    for i, point_on_curve in enumerate(points_on_curve):
        point_on_curve_tensor = tf.constant(point_on_curve, shape=(), dtype=tf.float32)
        model.point_on_curve.assign(point_on_curve_tensor)

        if len(points_on_curve) > 1:
            parameters = model.get_weighted_parameters(point_on_curve_tensor)
            if previous_parameters is not None:
                dl[i] = np.sqrt(np.sum(np.square(parameters - previous_parameters)))
            previous_parameters = parameters.copy()

        model.update_batchnorm(inputs=loaders["train"])

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

        train_predictions.append(train_results["pred"])
        train_output.append(train_results["output"])
        train_targets.append(train_results["target"])
        test_predictions.append(test_results["pred"])
        test_output.append(test_results["output"])
        test_targets.append(test_results["target"])

        train_accuracy_scores[i] = accuracy_score(
            train_results["target"], train_results["pred"]
        )
        train_f1_scores[i] = f1_score(
            train_results["target"], train_results["pred"], average="weighted"
        )
        test_accuracy_scores[i] = accuracy_score(
            test_results["target"], test_results["pred"]
        )
        test_f1_scores[i] = f1_score(
            test_results["target"], test_results["pred"], average="weighted"
        )

        values = [
            point_on_curve,
            train_losses[i],
            train_accuracy_scores[i] * 100,
            train_f1_scores[i],
            test_losses[i],
            test_accuracy_scores[i] * 100,
            test_f1_scores[i],
        ]

        print_stats_of_point_on_curve(values, i)

    if args.save_evaluation == True:
        save_stats_of_points_on_curve(
            train_losses,
            train_accuracy_scores,
            train_f1_scores,
            test_losses,
            test_accuracy_scores,
            test_f1_scores,
            points_on_curve,
            dl,
            args.dir,
            args.file_name_appendix,
        )

        save_predictions_and_probabilites(
            train_predictions,
            train_output,
            train_targets,
            test_predictions,
            test_output,
            test_targets,
            points_on_curve,
            args.dir,
            args.file_name_appendix,
        )

    if len(points_on_curve) > 1:
        print_and_save_summary_stats(
            train_losses,
            train_accuracy_scores,
            train_f1_scores,
            test_losses,
            test_accuracy_scores,
            test_f1_scores,
            points_on_curve,
            dl,
            args.dir,
            args.file_name_appendix,
            args.save_evaluation,
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
        Dict[str, tf.Tensor]: Evaluation statistics.
    """
    loss_sum = 0.0

    pred = []
    output = []
    target_list = []

    for input, target in test_loader:
        loss_batch, pred_batch, output_batch, target_batch = evaluate_batch(
            input=input, target=target, model=model, criterion=criterion
        )
        loss_sum += loss_batch
        pred += pred_batch
        output += output_batch
        target_list += target_batch

    return {
        "loss": loss_sum / n_test,
        "pred": np.array(pred),
        "output": np.array(output),
        "target": np.array(target_list),
    }


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
        float, list: Evaluation statistics.
    """
    output = model(input, training=False)
    loss = criterion(target, output)
    loss += tf.add_n(model.losses)

    loss = loss.numpy() * len(input)
    pred = tf.math.argmax(output, axis=1, output_type=tf.dtypes.int64)

    return loss, pred.numpy().tolist(), output.numpy().tolist(), target.numpy().tolist()


def print_stats_of_point_on_curve(values, i):
    """
    Displays relevant statistics of an epoch.

    Args:
        values (List): Statistics to be displayed.
        epoch (int): Current epoch.
    """
    columns = [
        "Point on curve",
        "Train loss",
        "Train accuracy (%)",
        "Train F1 score",
        "Test loss",
        "Test accuracy (%)",
        "Test F1 score",
    ]
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="10.4f")
    if i % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)


def save_stats_of_points_on_curve(
    train_losses,
    train_accuracy_scores,
    train_f1_scores,
    test_losses,
    test_accuracy_scores,
    test_f1_scores,
    points_on_curve,
    dl,
    dir: str,
    file_name_appendix: str,
):
    """
    Save relevants statistics of points on curve.

    Args:
        train_losses (numpy.ndarray): Array of train losses.
        train_accuracy_scores (numpy.ndarray): Array of train accuracy scores.
        train_f1_scores (numpy.ndarray): Array of train F1 scores.
        test_losses (numpy.ndarray): Array of test losses.
        test_accuracy_scores (numpy.ndarray): Array of test accuracy scores.
        test_f1_scores (numpy.ndarray): Array of test F1 scores.
        points_on_curve (numpy.ndarray): Array of points on curve evaluated.
        dl (numpy.ndarray): Norm of change of parameters between points on curve.
        dir (str): Directory to store file.
        file_name (str, optional): Name of file. Defaults to 'stats_of_points_on_curve.npz'.
    """
    file_name = "stats_of_points_on_curve" + file_name_appendix + ".npz"
    os.makedirs(dir, exist_ok=True)
    np.savez(
        os.path.join(dir, file_name),
        points_on_curve=points_on_curve,
        train_losses=train_losses,
        train_accuracy_scores=train_accuracy_scores,
        train_f1_scores=train_f1_scores,
        test_losses=test_losses,
        test_accuracy_scores=test_accuracy_scores,
        test_f1_scores=test_f1_scores,
        dl=dl,
    )


def save_predictions_and_probabilites(
    train_predictions,
    train_output,
    train_targets,
    test_predictions,
    test_output,
    test_targets,
    points_on_curve,
    dir: str,
    file_name_appendix: str,
):
    """
    Save relevant predictions and output of points on curve.

    Args:
        train_predictions (List): List of train prediction for points on curve evaluated.
        train_output (List): List of train output for points on curve evaluated.
        train_targets (List): List of train targets for points on curve evaluated.
        test_predictions (List): List of test prediction for points on curve evaluated.
        test_output (List): List of test output for points on curve evaluated.
        test_targets (List): List of test targets for points on curve evaluated.
        points_on_curve (numpy.ndarray): Array of points on curve evaluated.
        dir (str): Directory to store file.
        file_name (str, optional): Name of file. Defaults to 'predictions_and_probabilities_curve.npz'.
    """
    file_name = "predictions_and_probabilities_curve" + file_name_appendix + ".npz"
    os.makedirs(dir, exist_ok=True)
    np.savez(
        os.path.join(dir, file_name),
        points_on_curve=points_on_curve,
        train_predictions=train_predictions,
        train_targets=train_targets,
        train_output=train_output,
        test_predictions=test_predictions,
        test_targets=test_targets,
        test_output=test_output,
    )


def compute_stats(values, dl):
    """Compute summary statistics over all points on curve evaluated.

    Args:
        values (List): Statistics to be evaluated.
        dl (numpy.ndarray): Norm of change of parameters between points on curve.

    Returns:
        float: Summary statistics over all points on curve evaluated.
    """
    min = np.min(values)
    max = np.max(values)
    avg = np.mean(values)
    int = np.sum(0.5 * (values[:-1] + values[1:]) * dl[1:]) / np.sum(
        dl[1:]
    )  # What does this stats mean?
    return min, max, avg, int


def print_and_save_summary_stats(
    train_losses,
    train_accuracy_scores,
    train_f1_scores,
    test_losses,
    test_accuracy_scores,
    test_f1_scores,
    points_on_curve,
    dl,
    dir: str,
    file_name_appendix: str,
    save: bool = True,
):
    """Save summary statistics over all points on curve evaluated.

    Args:
        train_predictions (List): List of train prediction for points on curve evaluated.
        train_output (List): List of train output for points on curve evaluated.
        train_targets (List): List of train targets for points on curve evaluated.
        test_predictions (List): List of test prediction for points on curve evaluated.
        test_output (List): List of test output for points on curve evaluated.
        test_targets (List): List of test targets for points on curve evaluated.
        points_on_curve (numpy.ndarray): Array of points on curve evaluated.
        dir (str): Directory to store file.
        file_name (str, optional): Name of file. Defaults to 'predictions_and_probabilities_curve.npz'.
        save (bool, optional): Whether to save summary statistics or not. Defaults to True.
    """
    file_name = "summary_stats_curve" + file_name_appendix + ".npz"
    train_loss_min, train_loss_max, train_loss_avg, train_loss_int = compute_stats(
        train_losses, dl
    )
    (
        train_accuracy_scores_min,
        train_accuracy_scores_max,
        train_accuracy_scores_avg,
        train_accuracy_scores_int,
    ) = compute_stats(train_accuracy_scores, dl)
    (
        train_f1_scores_min,
        train_f1_scores_max,
        train_f1_scores_avg,
        train_f1_scores_int,
    ) = compute_stats(train_f1_scores, dl)

    test_loss_min, test_loss_max, test_loss_avg, test_loss_int = compute_stats(
        test_losses, dl
    )
    (
        test_accuracy_scores_min,
        test_accuracy_scores_max,
        test_accuracy_scores_avg,
        test_accuracy_scores_int,
    ) = compute_stats(test_accuracy_scores, dl)
    (
        test_f1_scores_min,
        test_f1_scores_max,
        test_f1_scores_avg,
        test_f1_scores_int,
    ) = compute_stats(test_f1_scores, dl)

    print("Length: %.2f" % np.sum(dl))
    print(
        tabulate.tabulate(
            [
                [
                    "Train loss",
                    train_losses[0],
                    train_losses[-1],
                    train_loss_min,
                    train_loss_max,
                    train_loss_avg,
                    train_loss_int,
                ],
                [
                    "Train accuracy (%)",
                    train_accuracy_scores[0] * 100,
                    train_accuracy_scores[-1] * 100,
                    train_accuracy_scores_min * 100,
                    train_accuracy_scores_max * 100,
                    train_accuracy_scores_avg * 100,
                    train_accuracy_scores_int * 100,
                ],
                [
                    "Train F1 score",
                    train_f1_scores[0],
                    train_f1_scores[-1],
                    train_f1_scores_min,
                    train_f1_scores_max,
                    train_f1_scores_avg,
                    train_f1_scores_int,
                ],
                [
                    "Test loss",
                    test_losses[0],
                    test_losses[-1],
                    test_loss_min,
                    test_loss_max,
                    test_loss_avg,
                    test_loss_int,
                ],
                [
                    "Test accuracy (%)",
                    test_accuracy_scores[0] * 100,
                    test_accuracy_scores[-1] * 100,
                    test_accuracy_scores_min * 100,
                    test_accuracy_scores_max * 100,
                    test_accuracy_scores_avg * 100,
                    test_accuracy_scores_int * 100,
                ],
                [
                    "Test F1 score",
                    test_f1_scores[0],
                    test_f1_scores[-1],
                    test_f1_scores_min,
                    test_f1_scores_max,
                    test_f1_scores_avg,
                    test_f1_scores_int,
                ],
            ],
            ["", "start", "end", "min", "max", "avg", "int"],
            tablefmt="simple",
            floatfmt="10.4f",
        )
    )

    if save == True:
        os.makedirs(dir, exist_ok=True)
        np.savez(
            os.path.join(dir, file_name),
            points_on_curve=points_on_curve,
            train_loss_min=train_loss_min,
            train_loss_max=train_loss_max,
            train_loss_avg=train_loss_avg,
            train_loss_int=train_loss_int,
            train_accuracy_min=train_accuracy_scores_min,
            train_accuracy_max=train_accuracy_scores_max,
            train_accuracy_avg=train_accuracy_scores_avg,
            train_accuracy_int=train_accuracy_scores_int,
            train_f1_scores_min=train_f1_scores_min,
            train_f1_scores_max=train_f1_scores_max,
            train_f1_scores_avg=train_f1_scores_avg,
            train_f1_scores_int=train_f1_scores_int,
            test_loss_min=test_loss_min,
            test_loss_max=test_loss_max,
            test_loss_avg=test_loss_avg,
            test_loss_int=test_loss_int,
            test_accuracy_min=test_accuracy_scores_min,
            test_accuracy_max=test_accuracy_scores_max,
            test_accuracy_avg=test_accuracy_scores_avg,
            test_accuracy_int=test_accuracy_scores_int,
            test_f1_scores_min=test_f1_scores_min,
            test_f1_scores_max=test_f1_scores_max,
            test_f1_scores_avg=test_f1_scores_avg,
            test_f1_scores_int=test_f1_scores_int,
        )


if __name__ == "__main__":
    main()
