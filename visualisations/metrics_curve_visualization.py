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
stat2 = "tr_acc"
stat3 = "tr_pr"
title = "Train Metrics"


plt.plot(metrics["points_on_curve"], metrics[stat1], color = 'blue', label = stat1)
plt.plot(metrics["points_on_curve"], metrics[stat2], color = 'orange', label = stat2)
plt.plot(metrics["points_on_curve"], metrics[stat3], color = 'black', label = stat3)
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)
plt.legend()
plt.show()


# %%
stat1 = "te_f1"
stat2 = "te_acc"
stat3 = "te_pr"
title = "Test Metrics"


plt.plot(metrics["points_on_curve"], metrics[stat1], color = 'blue', label = stat1)
plt.plot(metrics["points_on_curve"], metrics[stat2], color = 'orange', label = stat2)
plt.plot(metrics["points_on_curve"], metrics[stat3], color = 'black', label = stat3)
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)
plt.legend()
plt.show()

# %%

