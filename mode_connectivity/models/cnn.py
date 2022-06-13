import tensorflow as tf
from typing import List

__all__ = [
    "CNN",
]


class CNNBase(tf.keras.Model):
    num_classes: int
    conv_part: tf.keras.Sequential
    fc_part: tf.keras.Sequential

    #Sources:   
    #https://www.tensorflow.org/tutorials/customization/custom_layers
    #https://www.tensorflow.org/api_docs/python/tf/keras/layers

    #Network Structure:
    #https://github.com/constantin-crailsheim/dnn-mode-connectivity/blob/master/models/basiccnn.py

    # Optional: Alternatively inherit from tf.keras.layers.Layer
    # https://stackoverflow.com/questions/55109696/tensorflow-difference-between-tf-keras-layers-layer-vs-tf-keras-model

    # Comment: In contrast to PyTorch there are no input dimensions required in Tensorflow.

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        self.conv_part = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters= 32, kernel_size=(3, 3), activation='relu', kernel_initializer= 'glorot_normal' , bias_initializer= 'zeros'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters= 64, kernel_size=(3, 3), activation='relu', kernel_initializer= 'glorot_normal' , bias_initializer= 'zeros'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters= 64, kernel_size=(3, 3), kernel_initializer= 'glorot_normal' , bias_initializer= 'zeros'),
            tf.keras.layers.Flatten()])

        self.fc_part = tf.keras.Sequential([
            tf.keras.layers.Dense(units= 64, activation='relu'),
            tf.keras.layers.Dense(units= 64, activation='relu'),
            tf.keras.layers.Dense(units= self.num_classes)]) 

    def call(self, inputs, training=None, mask=None): #TO DO: Typehints & Chech which arguments necessary
        return self.conv_part(self.fc_part(inputs))
        #TO DO: Check if x = x.view(x.size(0), -1) necessary 
        

class CNNCurve(tf.keras.Model):
    def __init__(self, num_classes: int, fix_points: List[bool]):
        super().__init__()
        self.num_classes = num_classes
        self.fix_points = fix_points
        # TO DO

    def call(self, inputs, coeffs_t, training=None, mask=None): #TO DO: Typehints
        # TO DO
        pass


class CNN:
    base = CNNBase
    curve = CNNCurve
    kwargs = {}