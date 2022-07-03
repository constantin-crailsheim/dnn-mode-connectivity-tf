# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import tensorflow as tf
import seaborn as sns

from mode_connectivity.models.mlp import MLP
from mode_connectivity.models.linreg import LinReg
from mode_connectivity.curves import curves
from mode_connectivity.curves.net import CurveNet

# %%
# def features(x):
#     return np.hstack([x[:, None] / 2.0, (x[:, None] / 2.0) ** 2])

def features(x):
    return np.hstack([x[:, None] / 2.0, (x[:, None] / 2.0) ** 2, (x[:, None] / 2.0) ** 3, (x[:, None] / 2.0) ** 4])


def load_model(path, architecture, curve, num_bends, wd, fix_start, fix_end, num_classes: int, input_shape):
    curve = getattr(curves, curve)
    model = CurveNet(
        num_classes=num_classes,
        num_bends=num_bends,
        weight_decay=wd,
        curve=curve,
        curve_model=architecture.curve,
        fix_start=fix_start,
        fix_end=fix_end,
        architecture_kwargs=architecture.kwargs,
    )

    model.build(input_shape=input_shape)
    model.load_weights(filepath=path)
    model.compile()

    return model

# %%
path = "../results/Regression_MLP/checkpoints_curve/model-weights-epoch20"

model = load_model(
    path=path,
    architecture = MLP,
    curve = "Bezier",
    num_bends=3,
    wd=1e-4,
    fix_start=True,
    fix_end=True,
    num_classes=10,
    input_shape=(None, 2)
)
# %%

path = "../results/Regression_LinReg/checkpoints_curve/model-weights-epoch10"

model = load_model(
    path=path,
    architecture = LinReg,
    curve = "Bezier",
    num_bends=3,
    wd=5e-4,
    fix_start=True,
    fix_end=True,
    num_classes=10,
    input_shape=(None, 4)
)


# %%
data = np.load("../datasets/data.npy")

x = data[:, 0]
x_lin = np.linspace(min(x), max(x), 100)
f_lin = features(x_lin)
dataset = tf.constant(f_lin)

# %%
# Single curve
point_on_curve = 0.3

with tf.device("/cpu:0"):
    point_on_curve_tensor = tf.constant(point_on_curve, shape = (1,), dtype = tf.float64)
prediction = model(dataset, point_on_curve).numpy()
sns.set_style('darkgrid')
palette = sns.color_palette('colorblind')
blue = sns.color_palette()[0]
red = sns.color_palette()[3]
plt.figure(figsize=(9., 7.))
plt.plot(data[:, 0], data[:, 1], "o", color=red, alpha=0.7, markeredgewidth=1., markeredgecolor="k")
plt.plot(x_lin, prediction, color=blue)
plt.title("Data", fontsize=16)

# %%
# Multiple curves
T = 10

sns.set_style('darkgrid')
palette = sns.color_palette('colorblind')
blue = sns.color_palette()[0]
red = sns.color_palette()[3]
plt.figure(figsize=(9., 7.))
plt.plot(data[:, 0], data[:, 1], "o", color=red, alpha=0.7, markeredgewidth=1., markeredgecolor="k")
plt.title("Data", fontsize=16)

points_on_curve = np.linspace(0.0, 1.0, T)
for i, point_on_curve in enumerate(points_on_curve):
    with tf.device("/cpu:0"):
        point_on_curve_tensor = tf.constant(point_on_curve, shape = (1,), dtype = tf.float64)
    prediction = model(dataset, point_on_curve).numpy()
    plt.plot(x_lin, prediction, color=blue)

# %%

f_lin.shape[1]
# %%
