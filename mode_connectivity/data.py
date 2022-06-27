import os

import numpy as np
import random
import math

import tensorflow as tf
import tensorflow_datasets as tfds


class Transforms:
    class mnist:
        class CNN:

            pass
            # train = keras.Sequential([
            #     layers.LayerNormalization(axis = [0,1], mean = [0, 0])
            # ])

            # test = keras.Sequential([
            #     layers.Normalization(axis = None)
            # ])

            # train = lambda X, y: (
            #     tf.image.random_flip_left_right(X, seed = global_seed),
            #     tf.image.random_crop(X, size = (24,24), seed = global_seed), # global_seed not specified yet. See issue #2
            #     tf.image.resize_with_crop_or_pad(X, target_height = 28, target_width= 28),
            #     tf.image.per_image_standardization(X)
            # )

            # test = lambda X, y: (
            #     tf.image.per_image_standardization(X)
            # )

            # train = tf.keras.Sequential([
            #    #keras.Input(),
            #    layers.RandomFlip(mode = "horizontal", seed = 1),  #global_seed -> dummy variable. Not specified yet. See issue #2
            #    layers.RandomCrop(height = 32, width = 32, seed = 1),
            #    layers.ZeroPadding2D(padding = (4,4)),
            #    layers.Normalization(axis = None, mean = [0.485, 0.456, 0.406], variance = [pow(0.229,2), pow(0.224,2), pow(0.225,2)])
            # ])

            # test = tf.keras.Sequential([
            #    #keras.Input(),
            #    layers.Normalization(axis = None, mean = [0.485, 0.456, 0.406], variance = [pow(0.229,2), pow(0.224,2), pow(0.225,2)])
            # ])


class RegressionData:
    class MLP:

        pass


def data_loaders(
    dataset: str,
    path: str,
    batch_size: int,
    num_workers: int,
    transform_name: str,
    use_test: bool = False,
    shuffle_train: bool = True,
):
    # ds = getattr(keras.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    # transform = getattr(getattr(Transforms, dataset), transform_name)
    # train_set = ds(path, train = True, download = True, transform  = transform.train)

    if dataset == "mnist":
        (train_set, test_set), ds_info = tfds.load(
            name=dataset,
            data_dir=path,
            batch_size=batch_size,
            download=True,
            split=["train", "test"],
            shuffle_files=shuffle_train,
            as_supervised=True,
            with_info=True,
        )

        standardization = lambda input, label: (tf.cast(input, tf.float32) / 255, label)

        train_set_loader = train_set.map(standardization).shuffle(ds_info.splits["train"].num_examples)
        test_set_loader = test_set.map(standardization).shuffle(ds_info.splits["test"].num_examples)

        num_classes = 10 # TODO Find number of train labels in train_set object

        n_train = ds_info.splits["train"].num_examples
        n_test = ds_info.splits["test"].num_examples
    
    elif dataset == "regression":
        data = np.load("datasets/data.npy")
        n = data.shape[0]
        x, y = data[:, 0], data[:, 1]
        y = y[:, None]
        f = features(x)

        index_train = random.sample(range(n), math.floor(0.8*n))
        index_test = list(set(range(n)).difference(set(index_train)))

        train_dataset = tf.data.Dataset.from_tensor_slices((f[index_train,:], y[index_train]))
        test_dataset = tf.data.Dataset.from_tensor_slices((f[index_test,:], y[index_test]))

        train_set_loader = train_dataset.shuffle(100).batch(batch_size)
        test_set_loader = test_dataset.batch(batch_size)

        num_classes = 10 # How to deal with regression data for classification model

        n_train = len(index_train)
        n_test = len(index_test)

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
    ) 

def features(x):
    return np.hstack([x[:, None] / 2.0, (x[:, None] / 2.0) ** 2])


    # Questions:
    # Do we need to augment the data?
    # How to include Transforms Class? Is it even necessary at the moment?

    # return{
    #     'train': tf.data.Dataset.from_tensor_slices(
    #         process(*train_set)).batch(batch_size).shuffle(len(train_set[0])).map(transform.train),
    #     'test': tf.data.Dataset.from_tensor_slices(
    #         process(*test_set)).batch(batch_size).map(transform.test)
    #     }, max(train_set.train_labels) +1
