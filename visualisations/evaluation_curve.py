# %%
import os
import matplotlib.pyplot as plt
import numpy as np

# %%

os.chdir("..")
 
# %%
path = "results/MNIST_BasicCNN/evaluation_curve/curve.npz"
stats = np.load(path)

# %%

stat = "tr_loss"
title = "Train loss"

plt.plot(stats["points_on_curve"], stats[stat])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)

# %%

stat = "tr_acc"
title = "Train accuracy"

plt.plot(stats["points_on_curve"], stats[stat])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)


# %%

stat = "te_nll"
title = "Test NLL"

plt.plot(stats["points_on_curve"], stats[stat])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)

# %%

stat = "te_acc"
title = "Test accuracy"

plt.plot(stats["points_on_curve"], stats[stat])
plt.title(title + " for points on curve", fontsize=12)
plt.xlabel("Point on curve")
plt.ylabel(title)
