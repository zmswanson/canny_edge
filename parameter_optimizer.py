from canny_edge import CannyEdgeDetector
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from tqdm import tqdm

# load all image (noisy_image_*.png) and ground truth (ground_truth_*.png) pairs from data/simulated/
images = []
ground_truths = []

for i in range(0,50):
    image = cv2.imread(f"data/simulated/noisy_image_{i}.png", cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.imread(f"data/simulated/binary_image_{i}.png", cv2.IMREAD_GRAYSCALE)
    images.append(image)
    ground_truths.append(ground_truth)

# Split the data into a training set and a testing set
images_train, images_test, ground_truth_train, ground_truth_test = train_test_split(
    images, ground_truths, test_size=0.2, random_state=42
)

param_grid = {
    'sigma': [3, 8, 21, 34],
    'kernel_size': [3, 5, 9, 13],
    'neighbor_depth': [1, 2, 3],
    'high_thresh': [0.15, 0.27, 0.39, 0.51, 0.67],
    'low_thresh': [0.05, 0.09, 0.13]
}

def auc_scorer(params, images, ground_truths):
    """
    Performs Canny Edge Detection on the images using the given parameters and calculates the AUC 
    score using the ground truth. The AUC scores are averaged and returned.
    """
    auc_scores = []
    for image, ground_truth in zip(images, ground_truths):
        canny = CannyEdgeDetector(image, **params)
        edges = canny.threshold_hysteresis()
        auc_scores.append(roc_auc_score(ground_truth.flatten(), edges.flatten()))
    return np.mean(auc_scores)

# Generate all possible combinations of parameters
param_combinations = list(ParameterGrid(param_grid))

# Iterate through parameter combinations and find the best one using the training set
best_auc = 0.0
best_params = None

for params in tqdm(param_combinations, desc="Optimizing Parameters", leave=False):
    auc = auc_scorer(params, images_train, ground_truth_train)
    
    if auc > best_auc:
        best_auc = auc
        best_params = params

print(f"Best AUC on Training Set: {best_auc}")
print(f"Best Parameters: {best_params}")

# Evaluate the performance on the testing set
test_auc = auc_scorer(best_params, images_test, ground_truth_test)
print(f"AUC on Testing Set: {test_auc}")