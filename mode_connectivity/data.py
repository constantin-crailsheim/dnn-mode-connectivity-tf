import os
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers

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
            
            
            #train = tf.keras.Sequential([
            #    #keras.Input(),
            #    layers.RandomFlip(mode = "horizontal", seed = 1),  #global_seed -> dummy variable. Not specified yet. See issue #2
            #    layers.RandomCrop(height = 32, width = 32, seed = 1),
            #    layers.ZeroPadding2D(padding = (4,4)),
            #    layers.Normalization(axis = None, mean = [0.485, 0.456, 0.406], variance = [pow(0.229,2), pow(0.224,2), pow(0.225,2)])
            #])

            #test = tf.keras.Sequential([
            #    #keras.Input(),
            #    layers.Normalization(axis = None, mean = [0.485, 0.456, 0.406], variance = [pow(0.229,2), pow(0.224,2), pow(0.225,2)])
            #])
        
            
class RegressionData:
    
       class MLP:

            pass  
           

def loaders(
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
        name = dataset,
        data_dir = path,
        batch_size = batch_size,
        download = True,
        split=['train', 'test'],
        shuffle_files=shuffle_train,
        as_supervised=True,
        with_info=True,
        )
        train_set = train_set

        standardization = lambda input, label: (
            tf.cast(input, tf.float32) / 255,
            label)

    else:   
        if use_test:
            print('You are going to tun models on the test set. Are you sure?')
            pass

        else:
            print("Using train (45000) + validation (5000)")
            
            pass
    
    return{
        'train': train_set.map(standardization).shuffle(ds_info.splits['test'].num_examples), 
        'test': test_set.map(standardization).shuffle(ds_info.splits['test'].num_examples)
    }, 10 # TODO Find number of train labels in train_set object


    # Questions:
    # Do we need to augment the data?
    # How to include Transforms Class? Is it even necessary at the moment?


    # return{
    #     'train': tf.data.Dataset.from_tensor_slices(
    #         process(*train_set)).batch(batch_size).shuffle(len(train_set[0])).map(transform.train),
    #     'test': tf.data.Dataset.from_tensor_slices(
    #         process(*test_set)).batch(batch_size).map(transform.test)
    #     }, max(train_set.train_labels) +1
