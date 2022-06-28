import matplotlib.pyplot as plt
import sklearn as sklearn
from sklearn.metrics import ConfusionMatrixDisplay

import tabulate
import tensorflow as tf
from keras.layers import Layer
from keras.optimizers import Optimizer

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

def prediction_epoch(
    test_loader: Iterable,
    model: Layer,
    dir: str,
    criterion: Callable,
    n_test: int,
    regularizer: Union[Callable, None] = None,
    nll_sum = 0.0,
    loss_sum = 0.0,
    correct = 0.0,
    **kwargs,
):

    # PyTorch: model.eval()
    for input, target in test_loader:
        nll_batch, loss_batch, correct_batch, pred_batch, output_batch, target_batch = prediction_batch(
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
        pred = pred_batch
        output = output_batch
        target  = target_batch

    print_and_save_model_predictions_and_plots(pred, output, target, dir = dir, save: bool = True)

    

def prediction_batch(
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
        loss = tf.reduce_sum(
            loss
        ).numpy()  # What exactly are we trying to add up here, see original code? Check with PyTorch Code.
        pred = tf.math.argmax(output, axis=1, output_type=tf.dtypes.int64)
        correct = tf.math.reduce_sum(
            tf.cast(tf.math.equal(pred, target), tf.float32)
        ).numpy()

    return nll, loss, correct, pred, output, target
    

def print_and_save_model_predictions_and_plots(pred, output, target, dir : str, save: bool = True):

    get_confusion_mat_per_epoch(pred = pred, target = target, dir = dir)

    if save == True:
        os.makedirs(dir, exist_ok=True)
        np.savez(
            os.path.join(dir, 'predictions.npz'),
            pred=pred,
            output=output,
            target=target,
        )


def get_confusion_mat_per_epoch(
    pred: tf.Tensor,
    target: tf.Tensor,
    dir: str,
    save: bool = True,
):
    ConfusionMatrixDisplay.from_predictions(
        target, pred)
    
    if save == True:
        os.makedirs(dir, exist_ok=True)
        plt.savefig(
            os.path.join(dir, 'confusion_mat.png')
        )
    return plt.show()