# %%

# Just for experimental purposes, can be deleted once model runs through.
import mode_connectivity.data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# %%
loaders, num_classes = data.loaders(
    dataset = "mnist",
    path = "datasets",
    batch_size = 128,
    num_workers = 4,
    transform_name = "CNN",
    use_test = False,
    shuffle_train = True,
)

# %%

for input, target in loaders['train'].take(1):
    print(tf.math.reduce_max(input[0]))

# %%

print(loaders)


# %%

for input, target in loaders['train'].take(1):
    for i in range(128):
        image = np.asarray(input[i]).squeeze()
        plt.imshow(image)
        plt.show()
        print(target[i])
    

# %%
