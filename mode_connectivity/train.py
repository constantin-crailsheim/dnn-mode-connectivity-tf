import time
from typing import Callable
from typing import Tuple

import tabulate
import tensorflow as tf

from mode_connectivity.argparser import parse_train_arguments
from mode_connectivity.data import loaders
from mode_connectivity.models.cnn import CNN
from mode_connectivity.models.mlp import MLP
from mode_connectivity.utils import (
    adjust_learning_rate,
    check_batch_normalization,
    l2_regularizer,
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
        curve=args.curve, architecture=architecture, num_classes=num_classes
    )

    # TODO: criterion = F.cross_entropy
    criterion = tf.nn.softmax_cross_entropy_with_logits
    # Check if correct function, takes labels of shape [nbatch, nclass], while F.cross_entropy()
    # takes labels of shape [nBatch]
    regularizer = None if not args.curve else l2_regularizer(args.wd)
    # TODO : optimizer = torch.optim.SGD(
    #     filter(lambda param: param.requires_grad, model.parameters()),
    #     lr=args.lr,
    #     momentum=args.momentum,
    #     weight_decay=args.wd if args.curve is None else 0.0,
    # )
    optimizer = tf.keras.optimizers.SGD(
        # TODO how can we fit equivalent of arg params in PyTorch
        # PyTorch: params=filter(lambda param: param.requires_grad, model.parameters()),
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        learning_rate = args.lr,
        momentum = args.momentum,
        # Weight decay added into models/mlp.py
        # https://stackoverflow.com/questions/55046234/sgd-with-weight-decay-parameter-in-tensorflow
        # https://d2l.ai/chapter_multilayer-perceptrons/weight-decay.html
        # PyTorch: weight_decay=args.wd if args.curve is None else 0.0,
    )
    # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD
    

    load_checkpoint()
    save_checkpoint()

    train()

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


def get_model(curve: str, architecture, num_classes: int):
    if not curve:
        return architecture.base(num_classes=num_classes, **architecture.kwargs)

    # TODO return curve model
    return None


def load_checkpoint():
    pass


def save_checkpoint():
    pass


def train(
    args,
    loaders,
    model,
    learning_rate_schedule,
    criterion,
    regularizer,
    optimizer,
    start_epoch,
    columns,
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


def train_epoch(train_loader: Callable, model, optimizer, criterion, regularizer = None, lr_schedule: Callable = None) -> dict:
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    # PyTorch: model.train()
    for iter, (input, target) in enumerate(train_loader):
        if callable(lr_schedule):
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        loss_sum , correct = train_batch(
            input = input,
            target = target,
            loss_sum = loss_sum,
            correct = correct,
            model = model,
            optimizer = optimizer,
            criterion = criterion,
            regularizer = None,
            lr_schedule = None
            )
    
    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }


def test_epoch(test_loader: Callable, model, criterion, regularizer=None, **kwargs):
    nll_sum = 0.0
    loss_sum = 0.0
    correct = 0.0

    # PyTorch: model.eval()
    for input, target in test_loader:
        nll_sum, loss_sum, correct = test_batch(
            input = input,
            target = target,
            nll_sum = nll_sum,
            correct = correct,
            test_loader = test_loader,
            model = model,
            criterion = criterion,
            regularizer = None,
            **kwargs
        )
    
    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }


def train_batch(
    input, 
    target, 
    loss_sum, 
    correct, 
    model, 
    optimizer, 
    criterion, 
    regularizer = None, 
    lr_schedule = None
) -> Tuple[float, float]:
    # TODO Allocate model to CPU or GPU
    # PyTorch:
    # if torch.cuda.is_available():
#       model.cuda()
    # else:
    #   device = torch.device('cpu')
    #   model.to(device)


    with tf.GradientTape() as tape:
        output = model(input)
        loss = criterion(output, target) + model.losses # + model.losses necessary?
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

    return loss_sum, correct # Do we need to return the model as well?

def test_batch(
    input,
    target,
    nll_sum,
    correct,
    test_loader,
    model,
    criterion,
    regularizer = None,
    **kwargs
) -> Tuple[float, float]:
    # TODO Allocate model to CPU or GPU

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
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }


def print_epoch_stats(values, columns, epoch, start_epoch):
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="9.4f")
    if epoch % 40 == 1 or epoch == start_epoch:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)


if __name__ == "__main__":
    main()
