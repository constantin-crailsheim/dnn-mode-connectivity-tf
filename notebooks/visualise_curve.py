# %%
import os
import matplotlib.pyplot as plt
import numpy as np

# %%
path = "../results/MNIST_BasicCNN/evaluation_curve/stats_of_points_on_curve_epoch0.npz"
stats_init = np.load(path)

path = "../results/MNIST_BasicCNN/evaluation_curve/stats_of_points_on_curve_epoch10.npz"
stats_curve = np.load(path)

# %%

stat = "train_losses"
title = "Train loss"

_, plot = plt.subplots(1)
plot.plot(stats_init["points_on_curve"], stats_init[stat])
plot.plot(stats_curve["points_on_curve"], stats_curve[stat])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)
plt.legend(['Linear initialization', 'Trained curve'])

# %%

stat = "train_accuracy_scores"
title = "Train accuracy score"

_, plot = plt.subplots(1)
plot.plot(stats_init["points_on_curve"], stats_init[stat])
plot.plot(stats_curve["points_on_curve"], stats_curve[stat])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)
plt.legend(['Linear initialization', 'Trained curve'])

# %%

stat = "train_accuracy_scores"
title = "Train accuracy score"

_, plot = plt.subplots(1)
plot.plot(stats_init["points_on_curve"], stats_init[stat])
plot.plot(stats_curve["points_on_curve"], stats_curve[stat])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)
plt.legend(['Linear initialization', 'Trained curve'])

# %%

stat = "train_f1_scores"
title = "Train F1 scores"

_, plot = plt.subplots(1)
plot.plot(stats_init["points_on_curve"], stats_init[stat])
plot.plot(stats_curve["points_on_curve"], stats_curve[stat])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)
plt.legend(['Linear initialization', 'Trained curve'])

# %%

stat = "test_losses"
title = "Test loss"

_, plot = plt.subplots(1)
plot.plot(stats_init["points_on_curve"], stats_init[stat])
plot.plot(stats_curve["points_on_curve"], stats_curve[stat])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)
plt.legend(['Linear initialization', 'Trained curve'])

# %%

stat = "test_accuracy_scores"
title = "Test accuracy score"

_, plot = plt.subplots(1)
plot.plot(stats_init["points_on_curve"], stats_init[stat])
plot.plot(stats_curve["points_on_curve"], stats_curve[stat])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)
plt.legend(['Linear initialization', 'Trained curve'])

# %%

stat = "test_f1_scores"
title = "Test F1 score"

_, plot = plt.subplots(1)
plot.plot(stats_init["points_on_curve"], stats_init[stat])
plot.plot(stats_curve["points_on_curve"], stats_curve[stat])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)
plt.legend(['Linear initialization', 'Trained curve'])