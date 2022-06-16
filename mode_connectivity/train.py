import time
from typing import Callable, Dict, Iterable, Tuple, Union

import tabulate
import tensorflow as tf
from keras.layers import Layer
from keras.optimizers import Optimizer

# No mode_connectivity. needed to add before, since we are in the same folder.
from argparser import Arguments, parse_train_arguments
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
)
import curves

COLUMNS = ["ep", "lr", "tr_loss", "tr_acc", "te_nll", "te_acc", "time"]


def main():
    args = parse_train_arguments()

    # TODO: Set backends cudnnn
    set_seeds(seed=args.seed)

    loaders, num_classes, n_datasets = data_loaders(
        dataset=args.dataset,
        path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform_name=args.transform,
        use_test=args.use_test,
    )
    architecture = get_architecture(model_name=args.model)
    model = get_model(
        architecture = architecture,
        args = args,
        num_classes = num_classes,
    )

    # criterion = tf.nn.sparse_softmax_cross_entropy_with_logits
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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
        start_epoch = load_checkpoint(
            checkpoint_path=args.resume, model=model, optimizer=optimizer
        )
    save_checkpoint(directory=args.dir, epoch=start_epoch - 1, model=model, optimizer=optimizer)

    train(
        args=args,
        loaders=loaders,
        model=model,
        criterion=criterion,
        regularizer=regularizer,
        optimizer=optimizer,
        start_epoch=start_epoch,
        n_datasets=n_datasets,
    )


def set_seeds(seed: int):
    tf.random.set_seed(seed)
    # TODO torch.cuda.manual_seed(args.seed)


def get_architecture(model_name: str):
    if model_name == "CNN":
        return CNN
    if model_name == "MLP":
        return MLP
    raise KeyError(f"Unkown model {model_name}")


def get_model(architecture, args, num_classes: int): 
    #Changed method arguments to take args as input () since many of those variables needed in curve-case

    #If no curve is to be fit the base version of the architecture is initialised (e.g CNNBase instead of CNNCurve).
    if not args.curve:
        model = architecture.base(num_classes=num_classes, weight_decay= args.wd, **architecture.kwargs)
        return model

    #Otherwise the curve version of the architecture (e.g. CNNCurve) is initialised in the context of a CurveNet.
    #The CurveNet additionally contains the curve (e.g. Bezier) and imports the parameters of the pre-trained base-nets that constitute the outer points of the curve.
    else:
        pass
        curve = getattr(curves, args.curve)
        model = curves.CurveNet(
            num_classes,
            curve,
            architecture.curve,
            args.num_bends,
            args.fix_start,
            args.fix_end,
            architecture_kwargs=architecture.kwargs,
        )

        base_model = None
        if args.resume is None:
            for path, k in [(args.init_start, 0), (args.init_end, args.num_bends - 1)]:
                if path is not None:
                    if base_model is None:
                        base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
                    print('Loading %s as point #%d' % (path, k))

                    # Pytorch  Version:
                    # checkpoint = torch.load(path) #Angepasst                    
                    # base_model.load_state_dict(checkpoint['model_state']) #Angepasst

                    checkpoint = tf.train.Checkpoint(base_model)
                    checkpoint.restore(args.init_end) #Restores checkpoint in base_model
                    model.import_base_parameters(base_model, k)
            if args.init_linear:
                print('Linear initialization.')
                model.init_linear()
                
        return model


def train(
    args: Arguments,
    loaders: Dict[str, Iterable],
    model: Layer,
    criterion: Callable,
    regularizer: Union[Callable, None],
    optimizer: Optimizer,
    start_epoch: int,
    n_datasets: Dict,
):
    has_batch_normalization = check_batch_normalization(
        model
    )  # Not implemented yet, returns always True
    test_results = {"loss": None, "accuracy": None, "nll": None}

    for epoch in range(start_epoch, args.epochs + 1):
        time_epoch = time.time()

        lr = learning_rate_schedule(args.lr, epoch, args.epochs)
        # Not implemented yet:
        adjust_learning_rate(optimizer, lr)

        train_results = train_epoch(
            loaders["train"],
            model,
            optimizer,
            criterion,
            n_datasets["train"],
            regularizer,
        )

        if not args.curve or not has_batch_normalization:
            test_results = test_epoch(
                loaders["test"], model, criterion, n_datasets["test"], regularizer
            )

        if epoch % args.save_freq == 0:
            save_checkpoint(directory=args.dir, epoch=epoch, model=model, optimizer=optimizer)

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

        print_epoch_stats(values, epoch, start_epoch)

    if args.epochs % args.save_freq != 0:
        # Save last checkpoint if not already saved
        save_checkpoint(directory=args.dir, epoch=epoch, model=model, optimizer=optimizer)


def train_epoch(
    train_loader: Iterable,
    model: Layer,
    optimizer: Optimizer,
    criterion: Callable,
    n_train: int,
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
        loss_batch, correct_batch = train_batch(
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
        loss_sum += loss_batch
        correct += correct_batch

    return {
        "loss": loss_sum / n_train,  # Add function to find length
        "accuracy": correct * 100.0 / n_train,  # Add function to find length of dataset,
    }


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

    with tf.device("/cpu:0"):
        with tf.GradientTape() as tape:
            output = model(input)
            loss = criterion(target, output)  # + model.losses necessary?
            # PyTorch:
            # if regularizer is not None:
            #     loss += regularizer(model)
        grads = tape.gradient(loss, model.trainable_variables)
        grads_and_vars = zip(grads, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars)

        # See for above:
        # https://medium.com/analytics-vidhya/3-different-ways-to-perform-gradient-descent-in-tensorflow-2-0-and-ms-excel-ffc3791a160a
        # https://d2l.ai/chapter_multilayer-perceptrons/weight-decay.html (4.5.4)

        # What exactly are we trying to add up here, see original code? Check with PyTorch Code.
        loss = tf.reduce_sum(loss).numpy()
        pred = tf.math.argmax(output, axis=1, output_type=tf.dtypes.int64)
        # Is there an easier way?
        correct = tf.math.reduce_sum(tf.cast(tf.math.equal(pred, target), tf.float32)).numpy()

    return loss, correct  # Do we need to return the model as well?


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
        correct = tf.math.reduce_sum(tf.cast(tf.math.equal(pred, target), tf.float32)).numpy()

    return nll, loss, correct


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
