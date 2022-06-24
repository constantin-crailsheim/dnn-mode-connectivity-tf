import time
from typing import Callable, Dict, Iterable, Tuple, Union

import tabulate
import tensorflow as tf
import numpy as np
from keras.layers import Layer
from keras.optimizers import Optimizer

from mode_connectivity.curves import curves, layers, net

from argparser import Arguments, parse_evaluate_arguments
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
        num_classes=num_classes
    )
            
        
    model.compile()

    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    regularizer = None if not args.curve else l2_regularizer(args.wd)

    T = args.num_points
    ts = np.linspace(0.0, 1.0, T)
    train_loss = np.zeros(T)
    train_nll = np.zeros(T)
    train_accuracy = np.zeros(T)
    train_error = np.zeros(T)
    test_loss = np.zeros(T)
    test_nll = np.zeros(T)
    test_accuracy = np.zeros(T)
    test_error = np.zeros(T)
    dl = np.zeros(T)

    # previous_weights = None

    columns = ['t', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']

    
    for i, t_value in enumerate(ts):
        with tf.device("/cpu:0"):
            curve_point = tf.constant(t_value, shape = (1,), dtype = tf.float64)
        # TODO Check whether batch normilization is necessary.
        train_results = test_epoch(
                test_loader = loaders["train"],
                model = model,
                criterion = criterion,
                n_test = n_datasets["train"],
                curve_point = curve_point,
                regularizer = regularizer
            )
        test_results = test_epoch(
                test_loader = loaders["test"],
                model = model,
                criterion = criterion,
                n_test = n_datasets["test"],
                curve_point = curve_point,
                regularizer = regularizer
            )
        train_loss[i] = train_results['loss']
        train_nll[i] = train_results['nll']
        train_accuracy[i] = train_results['accuracy']
        train_error[i] = 100.0 - train_accuracy[i]
        test_loss[i] = test_results['loss']
        test_nll[i] = test_results['nll']
        test_accuracy[i] = test_results['accuracy']
        test_error[i] = 100.0 - test_results[i]

        values = [t_value, train_loss[i], train_nll[i], train_error[i], test_nll[i], test_error[i]]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
        if i % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)


def get_architecture(model_name: str):
    if model_name == "CNN":
        return CNN
    if model_name == "MLP":
        return MLP
    raise KeyError(f"Unkown model {model_name}")

def load_model(architecture, args: Arguments, num_classes: int):
    curve = getattr(curves, args.curve)
    model = tf.keras.models.load_model(args.ckpt)
        # custom_objects = {'CurveNet': net.CurveNet(num_classes, args.num_bends, args.wd, curve = curve, curve_model = architecture.curve, architecture_kwargs=architecture.kwargs)})
        # 'CurveLayer': layers.CurveLayer,
            # 'Conv2DCurve': layers.Conv2DCurve,
            # 'DenseCurve': layers.DenseCurve,
            # 'Curve': curves.Curve,
            # 'Bezier': curves.Bezier
    
    return model

def test_epoch(
    test_loader: Iterable,
    model: Layer,
    criterion: Callable,
    n_test: int,
    #curve_point: tf.Tensor,
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
            # curve_point = curve_point,
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
    #curve_point = tf.Tensor,
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

if __name__ == "__main__":
    main()