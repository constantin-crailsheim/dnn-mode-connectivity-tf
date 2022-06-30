# %%
import os
import matplotlib.pyplot as plt
import numpy as np

import sklearn as sklearn
from sklearn.metrics import ConfusionMatrixDisplay

# %%
# Execute only first time running the notebook.
os.chdir("..")

# %%
# Set point on curve

point_on_curve = 0.4

# %%
# Plot confusion matrix for train data

path = "results/MNIST_BasicCNN/evaluation_model/train_predictions_of_" + str(point_on_curve) + "_point_on_curve.npz"
stats = np.load(path)

predictions = stats['predictions']
target = stats['target']

ConfusionMatrixDisplay.from_predictions(target, predictions)

# %%
# Plot confusion matrix for test data

path = "results/MNIST_BasicCNN/evaluation_model/test_predictions_of_" + str(point_on_curve) + "_point_on_curve.npz"
stats = np.load(path)

predictions = stats['predictions']
target = stats['target']

ConfusionMatrixDisplay.from_predictions(target, predictions)
# %%
