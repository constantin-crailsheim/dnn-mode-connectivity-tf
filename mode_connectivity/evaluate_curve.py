import time
import os
from typing import Callable, Dict, Iterable, Tuple, Union

import tabulate
import tensorflow as tf
import numpy as np
from keras.layers import Layer

from mode_connectivity.curves import curves, layers, net
from argparser import Arguments, parse_evaluate_arguments
from mode_connectivity.curves.net import CurveNet
from data import data_loaders
from models.cnn import CNN
from models.mlp import MLP

from utils import (
    adjust_learning_rate,
    check_batch_normalization,
    l2_regularizer,
    learning_rate_schedule,
    load_checkpoint,
    save_checkpoint,
    save_model,
    save_weights,
)

def main():
    args = parse_evaluate_arguments()

    loaders, num_classes, n_datasets = data_loaders(
        dataset=args.dataset,
        path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform_name=args.transform,
        use_test=args.use_test,
    )
    architecture = get_architecture(model_name=args.model)
    model = load_model(
        architecture=architecture,
        args=args,
        num_classes=num_classes,
        input_shape=(None, 28, 28, 1),
    )

    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    regularizer = None if not args.curve else l2_regularizer(args.wd)

    T = args.num_points
    points_on_curve = np.linspace(0.0, 1.0, T)
    train_loss = np.zeros(T)
    train_nll = np.zeros(T)
    train_accuracy = np.zeros(T)
    train_error = np.zeros(T)
    test_loss = np.zeros(T)
    test_nll = np.zeros(T)
    test_accuracy = np.zeros(T)
    test_error = np.zeros(T)
    dl = np.zeros(T)

    previous_parameters = None
    
    for i, point_on_curve in enumerate(points_on_curve):
        with tf.device("/cpu:0"):
            point_on_curve_tensor = tf.constant(point_on_curve, shape = (1,), dtype = tf.float64)
        
        parameters = model.get_weighted_parameters(point_on_curve_tensor)
        if previous_parameters is not None:
             dl[i] = np.sqrt(np.sum(np.square(parameters - previous_parameters)))
        previous_parameters = parameters.copy()

        # TODO Check whether batch normilization is necessary.
        train_results = test_epoch(
                test_loader = loaders["train"],
                model = model,
                criterion = criterion,
                n_test = n_datasets["train"],
                point_on_curve = point_on_curve_tensor,
                regularizer = regularizer
            )
        test_results = test_epoch(
                test_loader = loaders["test"],
                model = model,
                criterion = criterion,
                n_test = n_datasets["test"],
                point_on_curve = point_on_curve_tensor,
                regularizer = regularizer
            )
        train_loss[i] = train_results['loss']
        train_nll[i] = train_results['nll']
        train_accuracy[i] = train_results['accuracy']
        train_error[i] = 100.0 - train_accuracy[i]
        test_loss[i] = test_results['loss']
        test_nll[i] = test_results['nll']
        test_accuracy[i] = test_results['accuracy']
        test_error[i] = 100.0 - test_accuracy[i]

        values = [point_on_curve, train_loss[i], train_nll[i], train_error[i], test_nll[i], test_error[i]]

        print_point_on_curve_stats(values, i)

    print_and_save_summary_stats(train_loss, train_nll, train_accuracy, train_error, test_loss, test_nll, test_accuracy, test_error, points_on_curve, dl, args.dir, save=True)
   

def get_architecture(model_name: str):
    if model_name == "CNN":
        return CNN
    if model_name == "MLP":
        return MLP
    raise KeyError(f"Unkown model {model_name}")

def load_model(architecture, args: Arguments, num_classes: int, input_shape):
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

def test_epoch(
    test_loader: Iterable,
    model: Layer,
    criterion: Callable,
    n_test: int,
    regularizer: Union[Callable, None] = None,
    **kwargs,
) -> Dict[str, float]:
    nll_sum = 0.0
    loss_sum = 0.0
    correct = 0.0

    # PyTorch: model.eval()
    for input, target in test_loader:
        nll_batch, loss_batch, correct_batch = test_batch(
            input=input,
            target=target,
            nll_sum=nll_sum,
            correct=correct,
            test_loader=test_loader,
            model=model,
            criterion=criterion,
            regularizer=regularizer,
            **kwargs,
        )
        nll_sum += nll_batch
        loss_sum += loss_batch
        correct += correct_batch

    return {
        "nll": nll_sum / n_test,  # Add function to find length
        "loss": loss_sum / n_test,  # Add function to find length
        "accuracy": correct * 100.0 / n_test,  # Add function to find length
    }

def test_batch(
    input: tf.Tensor,
    target: tf.Tensor,
    nll_sum: float,
    correct: float,
    test_loader: Iterable,
    model: Layer,
    criterion: Callable,
    regularizer: Union[Callable, None] = None,
    **kwargs,
) -> Dict[str, float]:
    # TODO Allocate model to GPU as well.

    with tf.device("/cpu:0"):
        output = model(inputs = input, **kwargs)
        nll = criterion(target, output)
        loss = tf.identity(nll)  # Correct funtion for nll.clone() in Pytorch
        # PyTorch:
        # if regularizer is not None:
        #     loss += regularizer(model)

        nll = tf.reduce_sum(nll).numpy()
        loss = tf.reduce_sum(
            loss
        ).numpy()  # What exactly are we trying to add up here, see original code? Check with PyTorch Code.
        pred = tf.math.argmax(output, axis=1, output_type=tf.dtypes.int64)
        correct = tf.math.reduce_sum(
            tf.cast(tf.math.equal(pred, target), tf.float32)
        ).numpy()

    return nll, loss, correct

def print_point_on_curve_stats(values, i):
    columns = ['Point on curve', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
    if i % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)


def stats(values, dl):
    min = np.min(values)
    max = np.max(values)
    avg = np.mean(values)
    int = np.sum(0.5 * (values[:-1] + values[1:]) * dl[1:]) / np.sum(dl[1:])
    return min, max, avg, int

def print_and_save_summary_stats(train_loss, train_nll, train_accuracy, train_error, test_loss, test_nll, test_accuracy, test_error, points_on_curve, dl, dir: str, save: bool = True):
    tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int = stats(train_loss, dl)
    tr_nll_min, tr_nll_max, tr_nll_avg, tr_nll_int = stats(train_nll, dl)
    tr_err_min, tr_err_max, tr_err_avg, tr_err_int = stats(train_error, dl)

    te_loss_min, te_loss_max, te_loss_avg, te_loss_int = stats(test_loss, dl)
    te_nll_min, te_nll_max, te_nll_avg, te_nll_int = stats(test_nll, dl)
    te_err_min, te_err_max, te_err_avg, te_err_int = stats(test_error, dl)

    print('Length: %.2f' % np.sum(dl))
    print(tabulate.tabulate([
            ['train loss', train_loss[0], train_loss[-1], tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int],
            ['train error (%)', train_error[0], train_error[-1], tr_err_min, tr_err_max, tr_err_avg, tr_err_int],
            ['test nll', test_nll[0], test_nll[-1], te_nll_min, te_nll_max, te_nll_avg, te_nll_int],
            ['test error (%)', test_error[0], test_error[-1], te_err_min, te_err_max, te_err_avg, te_err_int],
        ], [
            '', 'start', 'end', 'min', 'max', 'avg', 'int'
        ], tablefmt='simple', floatfmt='10.4f'))

    if save == True:
        os.makedirs(dir, exist_ok=True)
        np.savez(
            os.path.join(dir, 'curve.npz'),
            points_on_curve=points_on_curve,
            dl=dl,
            tr_loss=train_loss,
            tr_loss_min=tr_loss_min,
            tr_loss_max=tr_loss_max,
            tr_loss_avg=tr_loss_avg,
            tr_loss_int=tr_loss_int,
            tr_nll=train_nll,
            tr_nll_min=tr_nll_min,
            tr_nll_max=tr_nll_max,
            tr_nll_avg=tr_nll_avg,
            tr_nll_int=tr_nll_int,
            tr_acc=train_accuracy,
            tr_err=train_error,
            tr_err_min=tr_err_min,
            tr_err_max=tr_err_max,
            tr_err_avg=tr_err_avg,
            tr_err_int=tr_err_int,
            te_loss=test_loss,
            te_loss_min=te_loss_min,
            te_loss_max=te_loss_max,
            te_loss_avg=te_loss_avg,
            te_loss_int=te_loss_int,
            te_nll=test_nll,
            te_nll_min=te_nll_min,
            te_nll_max=te_nll_max,
            te_nll_avg=te_nll_avg,
            te_nll_int=te_nll_int,
            te_acc=test_accuracy,
            te_err=test_error,
            te_err_min=te_err_min,
            te_err_max=te_err_max,
            te_err_avg=te_err_avg,
            te_err_int=te_err_int,
        )


if __name__ == "__main__":
    main()