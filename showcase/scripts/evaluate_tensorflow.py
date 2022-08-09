import logging
import os
from typing import Dict, List

import tabulate

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tabulate
import tensorflow as tf

from ..argparser import Arguments, parse_evaluate_arguments
from ..logger import configure_loggers
from ..utils import disable_gpu, get_model_and_loaders, set_seeds

logger = logging.getLogger(__name__)


def evaluate(args: Arguments):
    configure_loggers()
    if args.disable_gpu:
        disable_gpu()

    logger.info("Starting evaluate")
    set_seeds(seed=args.seed)

    loaders, model = get_model_and_loaders(args)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(loss=loss, metrics=["accuracy"])
    # TODO Do we really want to evaluate on the train set?
    results = model.evaluate_points(
        loaders["test"],
        num_points=args.num_points,
        point_on_curve=args.point_on_curve,
        verbose=True,
        return_dict=True,
    )
    print_evaluation_results(results)


def print_evaluation_results(results: List[Dict[str, float]]):
    headers = ["Point on curve", "Loss", "Accuracy"]
    metrics = ["point_on_curve", "loss", "accuracy"]
    d = [list(f"{r[m]:.4f}" for m in metrics) for r in results]

    table = tabulate.tabulate(
        d,
        headers=headers,
        tablefmt="pretty",
    )
    print(table)


if __name__ == "__main__":
    args = parse_evaluate_arguments()
    evaluate(args=args)
