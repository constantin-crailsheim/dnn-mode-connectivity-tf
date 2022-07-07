# %%
import numpy as np
import sklearn as sklearn
from sklearn.metrics import ConfusionMatrixDisplay

# %%

def check_available_points_on_curve(dir: str, file_name: str):
    path = "../" + dir + file_name
    stats = np.load(path)
    return stats['points_on_curve']

def load_predictions(set, point_on_curve, dir: str, file_name: str):
    path = "../" + dir + file_name
    stats = np.load(path)
    id_point_on_curve = np.isclose(stats['points_on_curve'], point_on_curve)
    targets = stats[set + "_targets"][id_point_on_curve][0]
    predictions = stats[set + "_predictions"][id_point_on_curve][0]
    return targets, predictions


# %%

# Set directory and file

dir = "results/MNIST_BasicCNN/evaluation_curve/"

file_name = "predictions_and_probabilities_curve_epoch0.npz"

# %%

# Check which points of curve have been evaluated

print(check_available_points_on_curve(dir, file_name))

# %%
# Set evaluation parameters

set = "train"

point_on_curve = 0.5

# Plot confusion matrix
targets, predictions = load_predictions(set, point_on_curve, dir, file_name)

ConfusionMatrixDisplay.from_predictions(targets, predictions)
