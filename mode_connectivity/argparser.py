import argparse
from dataclasses import dataclass

import toml


@dataclass
class Arguments:
    # Config file
    config: str = None

    # Dataset
    dir: str = "results/"
    dataset: str = "mnist"
    data_path: str = "datasets/"

    # Model
    model: str = None
    curve: str = None
    ckpt: str = None
    save_freq: int = 50

    # Nodes initialization
    num_bends: int = 3
    init_start: str = None
    init_end: str = None
    fix_start: bool = False
    fix_end: bool = False
    init_linear: bool = True

    # Optimization
    batch_size: int = 128
    epochs: int = 200
    lr: float = 0.01
    momentum: float = 0.9
    wd: float = 1e-4
    resume_epoch: int = None

    # Computation
    seed: int = 1
    disable_gpu: bool = False

    # Evaluation:
    num_points: int = None
    point_on_curve: float = None
    save_evaluation: bool = True


def parse_config(config_name: str) -> Arguments:
    data = toml.load("config.toml")
    model_config = data.get(config_name, None)
    if not model_config:
        raise KeyError(f"Unknown model config {config_name}")
    model_config = {k.replace("-", "_"): v for k, v in model_config.items()}
    return Arguments(**model_config)


def parse_train_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description="DNN curve training")

    _add_config_argument(parser=parser)
    _add_dataset_arguments(parser=parser)
    _add_model_arguments(parser=parser)
    _add_nodes_init_arguments(parser=parser)
    _add_optimization_arguments(parser=parser)
    _add_computation_arguments(parser=parser)

    args = parser.parse_args()
    if args.config:
        return parse_config(args.config)
    return Arguments(**args.__dict__)


def parse_evaluate_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description="DNN evaluation")

    _add_config_argument(parser=parser)
    _add_dataset_arguments(parser=parser)
    _add_model_arguments(parser=parser)
    _add_nodes_init_arguments(parser=parser)
    _add_optimization_arguments(parser=parser)
    _add_computation_arguments(parser=parser)
    _add_evaluation_arguments(parser=parser)

    args = parser.parse_args()
    if args.config:
        return parse_config(args.config)
    return Arguments(**args.__dict__)


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="CONFIG",
        help="Name of config to use. If this is specified, other arguments are disregarded.",
    )


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dir",
        type=str,
        default="results/",
        help="training directory (default: /results/)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="dataset name (default: mnist)",
    )
    parser.add_argument(
        "--data-path",
        dest="data_path",
        type=str,
        default="datasets/",
        help="path to datasets location (default: None)",
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=True,
        help="model name (default: None)",
    )
    parser.add_argument(
        "--curve",
        type=str,
        default=None,
        help="curve type to use (default: None)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="checkpoint to load (default: None)",
    )
    parser.add_argument(
        "--save-freq",
        dest="save_freq",
        type=int,
        default=50,
        help="save frequency (default: 50)",
    )


def _add_nodes_init_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-bends",
        dest="num_bends",
        type=int,
        default=3,
        help="number of curve bends (default: 3)",
    )
    parser.add_argument(
        "--init-start",
        dest="init_start",
        type=str,
        default=None,
        help="path to SavedModel of init start point (default: None)",
    )
    parser.add_argument(
        "--fix-start",
        dest="fix_start",
        action="store_true",
        help="fix start point (default: off)",
    )
    parser.add_argument(
        "--init-end",
        dest="init_end",
        type=str,
        default=None,
        help="path to SavedModel of init end point (default: None)",
    )
    parser.add_argument(
        "--fix-end",
        dest="fix_end",
        action="store_true",
        help="fix end point (default: off)",
    )
    parser.add_argument(
        "--init-linear-off",
        dest="init_linear",
        action="store_false",
        help="turns off linear initialization of intermediate points (default: on)",
    )


def _add_optimization_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="input batch size (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-4,
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--resume-epoch",
        dest="resume_epoch",
        type=int,
        default=None,
        help="epoch to resume training from (default: None)",
    )


def _add_computation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--disable-gpu",
        dest="disable_gpu",
        action="store_true",
        help="disable GPU in computation (default: False)",
    )


def _add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-points",
        dest="num_points",
        type=int,
        default=None,
        help="number of points on the curve (default: None)",
    )
    parser.add_argument(
        "--point-on-curve",
        dest="point_on_curve",
        type=float,
        default=None,
        help="point on curve to be evaluated (default: None)",
    )
    parser.add_argument(
        "--save-evaluation",
        dest="save_evaluation",
        type=bool,
        default=True,
        help="Set whether evaluation should be saved",
    )
