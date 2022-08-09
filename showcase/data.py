import os

import numpy as np
import random
import math

import tensorflow as tf
import tensorflow_datasets as tfds


def data_loaders(
    dataset: str,
    path: str,
    batch_size: int
):
    """
    Prepares the data for training.

    Args:
        dataset (str): String indicating the type of the dataset. Either "mnist" for the CNN or "regression" for the MLP.
        path (str): Path to the dataset.
        batch_size (int): Amount of samples per batch.
        num_workers (int): Amount of workers.
        shuffle_train (bool, optional): Boolean indicating whether to shuffle the train set. Defaults to True.

    Returns:
        _type_: Tuple containing the data loaders and relevant variables e.g. train set size.
    """
    path = os.path.join(path, dataset.lower())

    if dataset == "mnist":
        (train_set, test_set), ds_info = tfds.load(
            name=dataset,
            data_dir=path,
            batch_size=batch_size,
            download=True,
            split=["train", "test"],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        standardization = lambda input, label: (tf.cast(input, tf.float32) / 255, label)

        train_set_loader = train_set.map(standardization).shuffle(ds_info.splits["train"].num_examples)
        test_set_loader = test_set.map(standardization).shuffle(ds_info.splits["test"].num_examples)

        num_classes = ds_info.features['label'].num_classes

        n_train = ds_info.splits["train"].num_examples
        n_test = ds_info.splits["test"].num_examples

        input_shape = (None, 28, 28, 1)
    
    elif dataset == "regression":
        data = np.load("datasets/data.npy")
        n = data.shape[0]
        x, y = data[:, 0], data[:, 1]
        y = y[:, None]

        def features(x):
            return np.hstack([x[:, None] / 2.0, (x[:, None] / 2.0) ** 2])
            
        f = features(x)

        index_train = random.sample(range(n), math.floor(0.8*n))
        index_test = list(set(range(n)).difference(set(index_train)))

        train_dataset = tf.data.Dataset.from_tensor_slices((f[index_train,:], y[index_train]))
        test_dataset = tf.data.Dataset.from_tensor_slices((f[index_test,:], y[index_test]))

        train_set_loader = train_dataset.shuffle(100).batch(batch_size)
        test_set_loader = test_dataset.batch(batch_size)

        num_classes = None

        n_train = len(index_train)
        n_test = len(index_test)

        input_shape = (None, f.shape[1])

    return (
        {
            "train": train_set_loader,
            "test": test_set_loader,
        },
        num_classes, 
        {
            "train": n_train,
            "test": n_test,
        }, 
        input_shape
    ) 