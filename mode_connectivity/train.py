import time
from typing import Callable, Dict, Iterable, Tuple, Union

import tabulate
import tensorflow as tf
from keras.layers import Layer
from keras.optimizers import Optimizer

from mode_connectivity.argparser import Arguments, parse_train_arguments
from mode_connectivity.data import loaders
from mode_connectivity.models.cnn import CNN
from mode_connectivity.models.mlp import MLP
from mode_connectivity.utils import (
    adjust_learning_rate,
    check_batch_normalization,
    l2_regularizer,
    learning_rate_schedule,
)

COLUMNS = ["ep", "lr", "tr_loss", "tr_acc", "te_nll", "te_acc", "time"]


def main():
    args = parse_train_arguments()

    # TODO: Set backends cudnnn
    set_seeds(seed=args.seed)

    loaders, num_classes = loaders(
        dataset=args.dataset,
        path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform_name=args.transform,
        use_test=args.use_test,
    )
    architecture = get_architecture(model_name=args.model)
    model = get_model(
        curve=args.curve, architecture=architecture, num_classes=num_classes, weight_decay=args.wd
    )

    criterion = tf.nn.softmax_cross_entropy_with_logits
    # TODO: Check if correct function, takes labels of shape [nbatch, nclass], while F.cross_entropy()
    # takes labels of shape [nBatch]
    regularizer = None if not args.curve else l2_regularizer(args.wd)
    optimizer = tf.keras.optimizers.SGD(
        # TODO how can we fit equivalent of arg params in PyTorch
        # PyTorch: params=filter(lambda param: param.requires_grad, model.parameters()),
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        learning_rate=args.lr,
        momentum=args.momentum,
        # Weight decay added into models/mlp.py
        # https://stackoverflow.com/questions/55046234/sgd-with-weight-decay-parameter-in-tensorflow
        # https://d2l.ai/chapter_multilayer-perceptrons/weight-decay.html
        # PyTorch: weight_decay=args.wd if args.curve is None else 0.0,
    )
    # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD

    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint()
    save_checkpoint()

    train(
        args=args,
        loaders=loaders,
        model=model,
        criterion=criterion,
        regularizer=regularizer,
        optimizer=optimizer,
        start_epoch=start_epoch,
    )

    if args.epochs % args.save_freq != 0:
        save_checkpoint()


def set_seeds(seed: int):
    pass


def get_architecture(model_name: str):
    if model_name == "CNN":
        return CNN
    if model_name == "MLP":
        return MLP
    raise KeyError(f"Unkown model {model_name}")


def get_model(curve: str, architecture, num_classes: int, weight_decay: float):
    if not curve:
        return architecture.base(num_classes=num_classes, weight_decay = weight_decay, **architecture.kwargs)

    # TODO return curve model
    return None


def load_checkpoint():
    # Pytorch:
    # start_epoch = 1
    # if args.resume is not None:
    #     print('Resume training from %s' % args.resume)
    #     checkpoint = torch.load(args.resume)
    #     start_epoch = checkpoint['epoch'] + 1
    #     model.load_state_dict(checkpoint['model_state'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state'])
    pass


def save_checkpoint():
    pass


def train(
    args: Arguments,
    loaders: Dict[str, Iterable],
    model: Layer,
    criterion: Callable,
    regularizer: Union[Callable, None],
    optimizer: Optimizer,
    start_epoch: int,
):
    has_batch_normalization = check_batch_normalization(model)
    test_results = {"loss": None, "accuracy": None, "nll": None}

    for epoch in range(start_epoch, args.epochs + 1):
        time_epoch = time.time()

        lr = learning_rate_schedule(args.lr, epoch, args.epochs)
        adjust_learning_rate(optimizer, lr)

        train_results = train_epoch(
            loaders["train"], model, optimizer, criterion, regularizer
        )

        if not args.curve or not has_batch_normalization:
            test_results = test_epoch(loaders["test"], model, criterion, regularizer)

        if epoch % args.save_freq == 0:
            save_checkpoint()

        time_epoch = time.time() - time_epoch
        values = [
            epoch,
            lr,
            train_results["loss"],
            train_results["accuracy"],
            test_results["nll"],
            test_results["accuracy"],
            time_epoch,
        ]

        print_epoch_stats()


def train_epoch(
    train_loader: Iterable,
    model: Layer,
    optimizer: Optimizer,
    criterion: Callable,
    regularizer: Union[Callable, None] = None,
    lr_schedule: Union[Callable, None] = None,
) -> Dict[str, float]:
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    # PyTorch: model.train()
    for iter, (input, target) in enumerate(train_loader):
        if callable(lr_schedule):
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        loss_sum, correct = train_batch(
            input=input,
            target=target,
            loss_sum=loss_sum,
            correct=correct,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            regularizer=regularizer,
            lr_schedule=lr_schedule,
        )

    return {
        "loss": loss_sum / len(train_loader.dataset),
        "accuracy": correct * 100.0 / len(train_loader.dataset),
    }


def test_epoch(
    test_loader: Iterable,
    model: Layer,
    criterion: Callable,
    regularizer: Union[Callable, None] = None,
    **kwargs,
) -> Dict[str, float]:
    nll_sum = 0.0
    loss_sum = 0.0
    correct = 0.0

    # PyTorch: model.eval()
    for input, target in test_loader:
        nll_sum, loss_sum, correct = test_batch(
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

    return {
        "nll": nll_sum / len(test_loader.dataset),
        "loss": loss_sum / len(test_loader.dataset),
        "accuracy": correct * 100.0 / len(test_loader.dataset),
    }


def train_batch(
    input: tf.Tensor,
    target: tf.Tensor,
    loss_sum: float,
    correct: float,
    model: Layer,
    optimizer: Optimizer,
    criterion: Callable,
    regularizer: Union[Callable, None] = None,
    lr_schedule: Union[Callable, None] = None,
) -> Tuple[float, float]:
    # TODO Allocate model to GPU as well, but no necessary at the moment, since we don't have GPUs.

    with tf.device('/cpu:0'):
        with tf.GradientTape() as tape:
            output = model(input)
            loss = criterion(output, target) + model.losses  # + model.losses necessary?
            # PyTorch:
            # if regularizer is not None:
            #     loss += regularizer(model)

        grads = tape.gradient(loss, model.trainable_variables)
        processed_grads = [g for g in grads]
        grads_and_vars = zip(processed_grads, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars)

        # See for above:
        # https://medium.com/analytics-vidhya/3-different-ways-to-perform-gradient-descent-in-tensorflow-2-0-and-ms-excel-ffc3791a160a
        # https://d2l.ai/chapter_multilayer-perceptrons/weight-decay.html (4.5.4)

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return loss_sum, correct  # Do we need to return the model as well?


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

    with tf.device('/cpu:0'):
        output = model(input, **kwargs)
        nll = criterion(output, target)
        loss = nll.clone()
        # PyTorch:
        # if regularizer is not None:
        #     loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        "nll": nll_sum / len(test_loader.dataset),
        "loss": loss_sum / len(test_loader.dataset),
        "accuracy": correct * 100.0 / len(test_loader.dataset),
    }


def print_epoch_stats(values, epoch, start_epoch):
    table = tabulate.tabulate([values], COLUMNS, tablefmt="simple", floatfmt="9.4f")
    if epoch % 40 == 1 or epoch == start_epoch:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)


if __name__ == "__main__":
    main()
