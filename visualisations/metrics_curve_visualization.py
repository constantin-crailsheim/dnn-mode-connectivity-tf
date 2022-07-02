# %%
import os
import matplotlib.pyplot as plt
import numpy as np

# %%
# Execute only first time running the notebook.
os.chdir("..")
# %%
path = "results/MNIST_BasicCNN/metrics_curve/preds_and_probs_curve.npz"
metrics = np.load(path)
# %%

stat1 = "tr_f1"
stat2 = "te_f1"
title = "Train vs. Test f1 score (weighted)"

_, plot = plt.subplots(1)
plot.plot(metrics["points_on_curve"], metrics[stat1])
plot.plot(metrics["points_on_curve"], metrics[stat2])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)

# %%
stat1 = "tr_acc"
stat2 = "te_acc"
title = "Train vs. Test accuracy (normalized)"

_, plot = plt.subplots(1)
plot.plot(metrics["points_on_curve"], metrics[stat1])
plot.plot(metrics["points_on_curve"], metrics[stat2])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)
# %%
# %%
stat1 = "tr_pr"
stat2 = "te_pr"
title = "Train vs. Test precision (weighted)"

_, plot = plt.subplots(1)
plot.plot(metrics["points_on_curve"], metrics[stat1])
plot.plot(metrics["points_on_curve"], metrics[stat2])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)
# %%