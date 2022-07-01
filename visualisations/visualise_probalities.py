# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Execute only first time running the notebook.
os.chdir("..")

# %%

def logistic(x):
    row_sums = np.sum(np.exp(x), axis = 1)
    return  np.exp(x)/row_sums[:,None]
# %%
# Set point on curve

point_on_curve = 0.4

# %%
# Prepare data for training set

path = "results/MNIST_BasicCNN/evaluation_model/train_predictions_of_" + str(point_on_curve) + "_point_on_curve.npz"
stats = np.load(path)

target = stats["target"]
output = stats['output']
output_probs = logistic(output)
output_prob_of_pred = output_probs[np.arange(output_probs.shape[0]), target]
data = pd.DataFrame({'Target': target, 'output_prob_of_pred': output_prob_of_pred})

# Prepare data for test set

path = "results/MNIST_BasicCNN/evaluation_model/train_predictions_of_" + str(point_on_curve) + "_point_on_curve.npz"
stats = np.load(path)

target = stats["target"]
output = stats['output']
output_probs = logistic(output)
output_prob_of_pred = output_probs[np.arange(output_probs.shape[0]), target]
data = pd.DataFrame({'Target': target, 'output_prob_of_pred': output_prob_of_pred})


# %%
# Evaluate training set

# target_to_evaluate = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

target_to_evaluate = [1, 7]

clip_min = 0
clip_max = 1

bandwidth = 0.01

data_subset = data[data["Target"].isin(target_to_evaluate)]

sns.kdeplot(data = data_subset, x = "output_prob_of_pred", hue = "Target", bw=bandwidth, clip = (clip_min, clip_max))
plt.xlabel('Probalities of predictions', fontsize=10)



# %%
