import time
import os
from typing import Callable, Dict, Iterable, Tuple, Union

import matplotlib.pyplot as plt
import sklearn as sklearn
from sklearn.metrics import ConfusionMatrixDisplay

import tabulate
import tensorflow as tf
import numpy as np
from keras.layers import Layer
from keras.optimizers import Optimizer

from argparser import Arguments, parse_evaluate_arguments
from data import data_loaders

import mode_connectivity.curves.curves as curves
from mode_connectivity.train import test_epoch
from mode_connectivity.train import test_batch
from mode_connectivity.argparser import Arguments, parse_train_arguments
from mode_connectivity.curves.net import CurveNet
from mode_connectivity.data import data_loaders
from mode_connectivity.models.cnn import CNN
from mode_connectivity.models.mlp import MLP
from mode_connectivity.utils import (
    adjust_learning_rate,
    check_batch_normalization,
    l2_regularizer,
    learning_rate_schedule,
    load_checkpoint,
    save_checkpoint,
    save_model,
    save_weights,
)

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

    with tf.device("/cpu:0"):
        point_on_curve_tensor = tf.constant(args.point_on_curve, shape = (1,), dtype = tf.float64)

    train_results = evaluation_epoch(
            test_loader = loaders["train"],
            model = model,
            criterion = criterion,
            n_test = n_datasets["train"],
            point_on_curve = point_on_curve_tensor,
            regularizer = regularizer
        )

    test_results = evaluation_epoch(
            test_loader = loaders["test"],
            model = model,
            criterion = criterion,
            n_test = n_datasets["test"],
            point_on_curve = point_on_curve_tensor,
            regularizer = regularizer
        )

    train_loss = train_results['loss']
    train_nll = train_results['nll']
    train_accuracy = train_results['accuracy']
    train_error = 100.0 - train_accuracy
    test_loss = test_results['loss']
    test_nll = test_results['nll']
    test_accuracy = test_results['accuracy']
    test_error = 100.0 - test_accuracy

    values = [args.point_on_curve, train_loss, train_nll, train_error, test_nll, test_error]
    print_model_stats(values)

    file_name_train = "train_predictions_of_" + str(args.point_on_curve) + "_point_on_curve.npz"
    save_predictions(train_results['pred'], train_results['output'], train_results['target'], dir = args.dir, file_name = file_name_train)

    file_name_test = "test_predictions_of_" + str(args.point_on_curve) + "_point_on_curve.npz"
    save_predictions(test_results['pred'], test_results['output'], test_results['target'], dir = args.dir, file_name = file_name_test)

def get_architecture(model_name: str):
    if model_name == "CNN":
        return CNN
    if model_name == "MLP":
        return MLP
    raise KeyError(f"Unkown model {model_name}")

def load_model(architecture, args: Arguments, num_classes: int, input_shape):
    # If no curve is to be fit the base version of the architecture is initialised (e.g CNNBase instead of CNNCurve).
    if not args.curve:
        model = architecture.base(
            num_classes=num_classes, weight_decay=args.wd, **architecture.kwargs
        )

        model.build(input_shape=input_shape)
        model.load_weights(filepath=args.ckpt)
        model.compile()

        return model

    # Otherwise the curve version of the architecture (e.g. CNNCurve) is initialised in the context of a CurveNet.
    # The CurveNet additionally contains the curve (e.g. Bezier) and imports the parameters of the pre-trained base-nets that constitute the outer points of the curve.
    else:
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


def evaluation_epoch(
    test_loader: Iterable,
    model: Layer,
    criterion: Callable,
    n_test: int,
    regularizer: Union[Callable, None] = None,
    nll_sum = 0.0,
    loss_sum = 0.0,
    correct = 0.0,
    **kwargs,
) -> Dict[str, tf.Tensor]:
    pred = []
    output = []
    target_list = []

    # PyTorch: model.eval()
    for input, target in test_loader:
        nll_batch, loss_batch, correct_batch, pred_batch, output_batch, target_batch = evaluation_batch(
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
        pred += pred_batch # Concatanate two lists
        output += output_batch
        target_list += target_batch

    return {
        "nll": nll_sum / n_test,
        "loss": loss_sum / n_test,
        "accuracy": correct * 100.0 / n_test,
        "pred": np.array(pred),  
        "output": np.array(output),  
        "target": np.array(target_list), 
    }
   
def evaluation_batch(
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
        output = model(input, **kwargs)
        nll = criterion(target, output)
        loss = tf.identity(nll)  # COrrect funtion for nll.clone() in Pytorch
        # PyTorch:
        # if regularizer is not None:
        #     loss += regularizer(model)

        nll = tf.reduce_sum(nll).numpy()
        loss = tf.reduce_sum(loss).numpy()
        pred = tf.math.argmax(output, axis=1, output_type=tf.dtypes.int64)
        correct = tf.math.reduce_sum(
            tf.cast(tf.math.equal(pred, target), tf.float32)
        ).numpy()

    return nll, loss, correct, pred.numpy().tolist(), output.numpy().tolist(), target.numpy().tolist()

def print_model_stats(values):
    columns = ['Point on curve', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
    table = table.split('\n')
    table = '\n'.join([table[1]] + table)
    print(table)

def save_predictions(pred, output, target, dir: str, file_name: str):
    os.makedirs(dir, exist_ok=True)
    np.savez(
        os.path.join(dir, file_name),
        predictions=pred,
        output=output,
        target=target,
    )

if __name__ == "__main__":
    main()