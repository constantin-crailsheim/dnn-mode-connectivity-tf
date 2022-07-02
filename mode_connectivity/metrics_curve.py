# %%
from re import I
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

import sklearn as sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

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
    train_predictions = list()
    train_target = list()
    train_probabilities = list()
    test_predictions = list()
    test_target = list()
    test_probabilities = list()
   # f1_score_values = np.zeros(T)
   # accuracy_score_values = np.zeros(T)
   # precision_score_values = np.zeros(T)
    train_f1_score_array = np.zeros(T)
    train_accuracy_score_values = np.zeros(T)
    train_precision_score_values = np.zeros(T)
    test_f1_score_array = np.zeros(T)
    test_accuracy_score_values = np.zeros(T)
    test_precision_score_values = np.zeros(T)
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
        
        train_predictions.append(train_results['pred'])
        train_probabilities.append(train_results['output'])
        train_target.append(train_results['target'])
        test_predictions.append(test_results['pred'])
        test_probabilities.append(test_results['output'])
        test_target.append(test_results['target'])
        train_f1_score_array[i] = f1_score(train_results['target'], train_results['pred'], average='weighted')
        train_accuracy_score_values[i] = accuracy_score(train_results['target'], train_results['pred'])
        train_precision_score_values[i] = precision_score(train_results['target'], train_results['pred'],  average='weighted')
        test_f1_score_array[i] = f1_score(test_results['target'], test_results['pred'], average='weighted')
        test_accuracy_score_values[i] = accuracy_score(test_results['target'], test_results['pred'])
        test_precision_score_values[i] = precision_score(test_results['target'], test_results['pred'],  average='weighted')

    save_preds_and_probs(train_predictions, train_probabilities, train_target, test_predictions, test_probabilities, test_target, train_f1_score_array, train_accuracy_score_values, train_precision_score_values, test_f1_score_array, test_accuracy_score_values, test_precision_score_values, points_on_curve, dl, args.dir, save=True)
   

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



def save_preds_and_probs(train_predictions, train_probabilities, train_target, test_predictions, test_probabilities, test_target, train_f1_score_array, train_accuracy_score_values, train_precision_score_values, test_f1_score_array, test_accuracy_score_values, test_precision_score_values, points_on_curve, dl, dir: str, save: bool = True):
 
    if save == True:
        os.makedirs(dir, exist_ok=True)
        np.savez(
            os.path.join(dir, 'preds_and_probs_curve.npz'),
            points_on_curve=points_on_curve,
            dl=dl,
            tr_preds=train_predictions,
            tr_target=train_target,
            tr_probs=train_probabilities,
            tr_f1=train_f1_score_array,
            tr_acc=train_accuracy_score_values,
            tr_pr=train_precision_score_values,
            te_preds_max=test_predictions,
            te_target=test_target,
            te_probs=test_probabilities,
            te_f1=test_f1_score_array,
            te_acc=test_accuracy_score_values,
            te_pr=test_precision_score_values
        )


if __name__ == "__main__":
    main()

# %%