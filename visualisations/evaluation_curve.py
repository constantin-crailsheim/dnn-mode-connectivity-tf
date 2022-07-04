# %%
import os
import matplotlib.pyplot as plt
import numpy as np

# %%
# Execute only first time running the notebook.
os.chdir("..")

# %%
path = "results/MNIST_BasicCNN/evaluation_curve/summary_stats_curve_epoch0.npz"
stats_init = np.load(path)

path = "results/MNIST_BasicCNN/evaluation_curve/summary_stats_curve_epoch10.npz"
stats_curve = np.load(path)


# %%

stat = "tr_loss"
title = "Train loss"

_, plot = plt.subplots(1)
plot.plot(stats_init["points_on_curve"], stats_init[stat])
plot.plot(stats_curve["points_on_curve"], stats_curve[stat])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)

# %%

stat = "tr_acc"
title = "Train accuracy"

_, plot = plt.subplots(1)
plot.plot(stats_init["points_on_curve"], stats_init[stat])
plot.plot(stats_curve["points_on_curve"], stats_curve[stat])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)


# %%

stat = "te_nll"
title = "Test NLL"

_, plot = plt.subplots(1)
plot.plot(stats_init["points_on_curve"], stats_init[stat])
plot.plot(stats_curve["points_on_curve"], stats_curve[stat])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)

# %%

stat = "te_acc"
title = "Test accuracy"

_, plot = plt.subplots(1)
plot.plot(stats_init["points_on_curve"], stats_init[stat])
plot.plot(stats_curve["points_on_curve"], stats_curve[stat])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)

# %%
