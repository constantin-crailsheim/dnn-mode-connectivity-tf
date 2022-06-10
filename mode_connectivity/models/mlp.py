import tensorflow as tf

__all__ = [
    "MLP",
]


class MLPBase:  # Inherit equivalent of torch.nn
    def __init__(self, num_classes):
        super(MLPBase, self).__init__()
        self.fc_part = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, 
                    activation="relu",
                    kernel_regularizer = tf.keras.regularizer.l2(args.wd if args.curve is None else 0.0)),
                tf.keras.layers.Dense(10,
                kernel_regularizer = tf.keras.regularizer.l2(args.wd if args.curve is None else 0.0)),
            ]
        )

    def forward(self, x):
        x = self.fc_part(x)
        return x


class MLPCurve:  # Inherit equivalent of torch.nn
    def __init__(self, num_classes, fix_points):
        super(MLPCurve, self).__init__()

    def forward(self, x, coeffs_t):
        return x


class MLP:
    base = MLPBase
    curve = MLPCurve
    kwargs = {}
