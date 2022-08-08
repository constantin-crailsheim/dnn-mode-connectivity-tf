from unittest import mock

import pytest

# Load files from parent directory
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from mode_connectivity.argparser import (
    Arguments,
    parse_train_arguments,
    parse_evaluate_arguments
)


BASIC_ARGS = ["python", "--model", "SomeModel"]


class TestArgparser:
    def test_parse_train_arguments_basic(self):
        with mock.patch("sys.argv", BASIC_ARGS):
            args = parse_train_arguments()
            assert isinstance(args, Arguments)
            assert args.__dict__.keys() == Arguments().__dict__.keys()
            assert args.model == "SomeModel"

    def test_parse_train_arguments_no_model(self):
        with mock.patch("sys.argv", ["python"]):
            with pytest.raises(SystemExit):
                parse_train_arguments()

    def test_parse_train_arguments_unknown_arg(self):
        with mock.patch("sys.argv", ["python", "--this-is-no-option", "NoOption"]):
            with pytest.raises(SystemExit):
                parse_train_arguments()

    def test_parse_dataset_arguments(self):
        dataset_args = [
            "--dir",
            "/path/to/some/dir",
            "--dataset",
            "SomeDataset",
            "--data-path",
            "SomePath",
        ]
        with mock.patch("sys.argv", BASIC_ARGS + dataset_args):
            args = parse_train_arguments()
            assert args.dir == "/path/to/some/dir"
            assert args.dataset == "SomeDataset"
            assert args.data_path == "SomePath"

    def test_parse_model_arguments_model(self):
        model_args = [
            "--curve",
            "SomeCurve",
            "--ckpt",
            "SomeCheckpoint",
            "--save-freq",
            "50"
        ]
        with mock.patch("sys.argv", BASIC_ARGS + model_args):
            args = parse_train_arguments()
            assert args.curve == "SomeCurve"
            assert args.ckpt == "SomeCheckpoint"
            assert args.save_freq == 50

    def test_parse_nodes_init_arguments_model(self):
        model_args = [
            "--num-bends",
            "9",
            "--init-start",
            "TheCheckpoint",
            "--init-end",
            "TheEnd",
        ]
        with mock.patch("sys.argv", BASIC_ARGS + model_args):
            args = parse_train_arguments()
            assert args.num_bends == 9
            assert args.init_start == "TheCheckpoint"
            assert args.fix_start is False
            assert args.init_end == "TheEnd"
            assert args.fix_end is False
            assert args.init_linear is True

    def test_parse_optimization_arguments_model(self):
        model_args = [
            "--batch-size",
            "99",
            "--epochs",
            "42",
            "--lr",
            "0.5",
            "--momentum",
            "1.0",
            "--wd",
            "0.01",
            "--resume-epoch",
            "20"
        ]
        with mock.patch("sys.argv", BASIC_ARGS + model_args):
            args = parse_train_arguments()
            assert args.batch_size == 99
            assert args.epochs == 42
            assert args.lr == 0.5
            assert args.momentum == 1.0
            assert args.wd == 0.01
            assert args.resume_epoch == 20

    def test_parse_computation_arguments(self):
        compute_args = ["--seed", "123"]
        with mock.patch("sys.argv", BASIC_ARGS + compute_args):
            args = parse_train_arguments()
            assert args.seed == 123
            assert args.disable_gpu is False

    def test_parse_evaluation_arguments_model(self):
        model_args = [
            "--num-points",
            "11",
            "--point-on-curve",
            "0.3",
        ]
        with mock.patch("sys.argv", BASIC_ARGS + model_args):
            args = parse_evaluate_arguments()
            assert args.num_points == 11
            assert args.point_on_curve == 0.3
            assert args.save_evaluation is True

    def test_parse_model_arguments_init_linear_off(self):
        model_args = ["--init-linear-off"]
        with mock.patch("sys.argv", BASIC_ARGS + model_args):
            args = parse_train_arguments()
            assert args.init_linear is False

    def test_parse_nodes_init_arguments_fix(self):
        checkpoint_args = ["--fix-start", "--fix-end"]
        with mock.patch("sys.argv", BASIC_ARGS + checkpoint_args):
            args = parse_train_arguments()
            assert args.fix_start is True
            assert args.fix_end is True
