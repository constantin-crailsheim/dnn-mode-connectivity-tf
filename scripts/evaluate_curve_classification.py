import os
from typing import Callable, Dict, Iterable, Tuple, Union

import numpy as np
import tabulate

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from keras.layers import Layer

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from mode_connectivity.argparser import Arguments, parse_evaluate_arguments
from mode_connectivity.data import data_loaders

from mode_connectivity.curves import curves
from mode_connectivity.curves.net import CurveNet

from mode_connectivity.models.cnn import CNN
from mode_connectivity.models.mlp import MLP

from mode_connectivity.utils import disable_gpu, get_model


def main():
    args = parse_evaluate_arguments()
    if args.disable_gpu:
        disable_gpu()

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
            save=args.save_evaluation,
        )


def get_architecture(model_name: str):
    if model_name == "CNN":
        return CNN
    if model_name == "MLP":
        return MLP
    raise KeyError(f"Unkown model {model_name}")


def load_model(
    architecture, args: Arguments, num_classes: Union[int, None], input_shape
):
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

    model.build(input_shape=input_shape)
    model.load_weights(filepath=args.ckpt)
    model.compile()

    return model


def evaluate_epoch(
    test_loader: Iterable, model: Layer, criterion: Callable, n_test: int
) -> Dict[str, tf.Tensor]:

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
    input: tf.Tensor, target: tf.Tensor, model: Layer, criterion: Callable
) -> Dict[str, float]:

    output = model(input, training=False)
    loss = criterion(target, output)

    loss = loss.numpy() * len(input)
    pred = tf.math.argmax(output, axis=1, output_type=tf.dtypes.int64)

    return loss, pred.numpy().tolist(), output.numpy().tolist(), target.numpy().tolist()


def print_stats_of_point_on_curve(values, i):
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
    file_name: str = "stats_of_points_on_curve.npz",
):

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
        dl=dl,  # TODO do we need to save this here?
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
    file_name: str = "predictions_and_probabilities_curve.npz",
):

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
    file_name: str = "summary_stats_curve.npz",
    save: bool = True,
):

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
