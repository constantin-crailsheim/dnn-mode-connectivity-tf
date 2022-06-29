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
path = "results/MNIST_BasicCNN/evaluation_model/train_predictions_of_0.0_point_on_curve.npz"
stats = np.load(path)

predictions = stats['predictions']
target = stats['target']

# %%

ConfusionMatrixDisplay.from_predictions(target, predictions)
