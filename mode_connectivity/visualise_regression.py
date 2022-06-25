# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import tensorflow as tf
import seaborn as sns

from mode_connectivity.models.mlp import MLP


# %%
def features(x):
    return np.hstack([x[:, None] / 2.0, (x[:, None] / 2.0) ** 2])

# %%
os.chdir("..")
data = np.load("datasets/data.npy")

# %%
architecture = MLP
input_shape = (None, 2)

path1 = "results/Regression_MLP/checkpoints_model_1/model-weights-epoch20"
path2 = "results/Regression_MLP/checkpoints_model_2/model-weights-epoch20"

base_model1 = architecture.base(num_classes=10, weight_decay=1e-4, **architecture.kwargs)
base_model1.build(input_shape=input_shape)
base_model1.load_weights(path1)

base_model2 = architecture.base(num_classes=10, weight_decay=1e-4, **architecture.kwargs)
base_model2.build(input_shape=input_shape)
base_model2.load_weights(path2)


# %%
x = data[:, 0]
x_lin = np.linspace(min(x), max(x), 100)
f_lin = features(x_lin)

dataset = tf.constant(f_lin)

prediction1 = base_model1(dataset).numpy()
prediction2 = base_model2(dataset).numpy()

# %%
sns.set_style('darkgrid')
palette = sns.color_palette('colorblind')
blue = sns.color_palette()[0]
red = sns.color_palette()[3]
plt.figure(figsize=(9., 7.))
plt.plot(data[:, 0], data[:, 1], "o", color=red, alpha=0.7, markeredgewidth=1., markeredgecolor="k")
plt.plot(x_lin, prediction1, color=blue)
plt.plot(x_lin, prediction2, color=red)
plt.title("Data", fontsize=16)



# %%
