import argparse
from dataclasses import dataclass


@dataclass
class Arguments:
    # Dataset
    dir: str = "/results/"
    dataset: str = "MNIST"
    use_test: bool = False
    data_path: str = "/datasets/"

    # Model
    model: str = None
    curve: str = None
    transform: str = "CNN"
    batch_size: int = 128
    num_bends: int = 3
    init_linear: bool = True
    epochs: int = 200
    lr: float = 0.01
    momentum: float = 0.9
    wd: float = 1e-4

    # Compute
    num_workers: int = 4
    seed: int = 1

    # Checkpoints
    init_start: str = None
    init_end: str = None
    fix_end: bool = False
    fix_start: bool = False
    resume: str = None
    save_freq: int = 50


def parse_train_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description="DNN curve training")

    _add_dataset_arguments(parser=parser)
    _add_compute_arguments(parser=parser)
    _add_model_arguments(parser=parser)
    _add_checkpoint_arguments(parser=parser)

    args = parser.parse_args()
    return Arguments(**args.__dict__)


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dir",
        type=str,
        default="/results/",
        metavar="DIR",
        help="training directory (default: /results/)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        metavar="DATASET",
        help="dataset name (default: MNIST)",
    )
    parser.add_argument(
        "--use-test",
        action="store_true",
        help="switches between validation and test set (default: validation)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/datasets/",
        metavar="PATH",
        help="path to datasets location (default: None)",
    )


def _add_compute_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        metavar="N",
        help="number of workers (default: 4)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL",
        required=True,
        help="model name (default: None)",
    )
    parser.add_argument(
        "--curve",
        type=str,
        default=None,
        metavar="CURVE",
        help="curve type to use (default: None)",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="CNN",
        metavar="TRANSFORM",
        help="transform name (default: CNN)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size (default: 128)",
    )
    parser.add_argument(
        "--num-bends",
        type=int,
        default=3,
        metavar="N",
        help="number of curve bends (default: 3)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-4,
        metavar="WD",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--init-linear-off",
        dest="init_linear",
        action="store_false",
        help="turns off linear initialization of intermediate points (default: on)",
    )


def _add_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--init-start",
        type=str,
        default=None,
        metavar="CKPT",
        help="checkpoint to init start point (default: None)",
    )
    parser.add_argument(
        "--fix-start",
        dest="fix_start",
        action="store_true",
        help="fix start point (default: off)",
    )
    parser.add_argument(
        "--init-end",
        type=str,
        default=None,
        metavar="CKPT",
        help="checkpoint to init end point (default: None)",
    )
    parser.add_argument(
        "--fix-end",
        dest="fix_end",
        action="store_true",
        help="fix end point (default: off)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CKPT",
        help="checkpoint to resume training from (default: None)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50,
        metavar="N",
        help="save frequency (default: 50)",
    )
