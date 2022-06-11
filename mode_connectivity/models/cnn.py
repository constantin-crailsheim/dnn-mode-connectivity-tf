import tensorflow as tf

__all__ = [
    "CNN",
]


class CNNBase:  # Inherit equivalent of torch.nn
    # -> Should be tf.keras.layers.Layer
    # https://stackoverflow.com/a/69151656/10429039
    def __init__(self, num_classes):
        super(CNNBase, self).__init__()
        self.conv_part = tf.keras.models.Sequential(tf.keras.layers())
        self.fc_part = tf.keras.models.Sequential()

    def forward(self, x):
        return x


class CNNCurve:  # Inherit equivalent of torch.nn
    def __init__(self, num_classes, fix_points):
        super(CNNCurve, self).__init__()

    def forward(self, x, coeffs_t):
        return x


class CNN:
    base = CNNBase
    curve = CNNCurve
    kwargs = {}
