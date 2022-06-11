import tensorflow as tf

__all__ = [
    "CNN",
]


class CNNBase(tf.keras.Model):
    #Sources:   
    #https://www.tensorflow.org/tutorials/customization/custom_layers
    #https://www.tensorflow.org/api_docs/python/tf/keras/layers

    #Network Structure:
    #https://github.com/constantin-crailsheim/dnn-mode-connectivity/blob/master/models/basiccnn.py

    # Optional: Alternatively inherit from tf.keras.layers.Layer
    # https://stackoverflow.com/questions/55109696/tensorflow-difference-between-tf-keras-layers-layer-vs-tf-keras-model

    # Comment: In contrast to PyTorch there are no input dimensions required in Tensorflow.

    def __init__(self, num_classes):
        super(CNNBase, self).__init__()
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

    def call(self, x):
        # Forward compuation (Equivalent to forward() in PyTorch)
        return self.conv_part(self.fc_part(x))
        

class CNNCurve:  # Inherit equivalent of torch.nn
    def __init__(self, num_classes, fix_points):
        super(CNNCurve, self).__init__()
        self.num_classes = num_classes
        self.fix_points = fix_points

        # TO DO

    def forward(self, x, coeffs_t):
        # TO DO
        pass


class CNN:
    base = CNNBase
    curve = CNNCurve
    kwargs = {}