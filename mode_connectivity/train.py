import time

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
    regularizer = None if not args.curve else l2_regularizer(args.wd)
    # TODO : optimizer = torch.optim.SGD(
    #     filter(lambda param: param.requires_grad, model.parameters()),
    #     lr=args.lr,
    #     momentum=args.momentum,
    #     weight_decay=args.wd if args.curve is None else 0.0,
    # )

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


def train_epoch():
    pass


def test_epoch():
    pass


def train_batch():
    pass


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
