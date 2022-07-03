# %%
import os
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%

def softmax(x):
    row_sums = np.sum(np.exp(x), axis = 1)
    return  np.exp(x)/row_sums[:,None]

def load_probabilities(set, point_on_curve):
    path = "../results/MNIST_BasicCNN/evaluation_model/" + set + "_predictions_of_" + str(point_on_curve) + "_point_on_curve.npz"
    stats = np.load(path)
    target = stats["target"]
    output = stats['output']
    output_probs = softmax(output)
    output_prob_of_pred = output_probs[np.arange(output_probs.shape[0]), target]
    data = pd.DataFrame({'Target': target, 'output_prob_of_pred': output_prob_of_pred})
    return data

# %%

# Set evaluation parameters

set = "train" # "train" or "test"

point_on_curve = 0.4

target_to_evaluate = [1, 7] # Choose from [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

clip_min = 0.99 # Minimum of x-axis (probabilites)
clip_max = 1 # Maximsum of x-axis (probabilites)

bandwidth = 0.01 # Bandwidth of kernel for densities

# Load dataset

data = load_probabilities(set, point_on_curve)
data_subset = data[data["Target"].isin(target_to_evaluate)]

# Plot density plot
sns.kdeplot(data = data_subset, x = "output_prob_of_pred", hue = "Target", bw_method=bandwidth, clip = (clip_min, clip_max))
plt.xlabel('Probalities of predictions', fontsize=10)

# Ideally we would see:
# Very high density for probabilites close to 1
# Very low density for probabilities lower than 1
# Not much deviation in probabilites between different targets

