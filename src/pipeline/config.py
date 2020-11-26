import os
from enum import Enum
import numpy as np
import time
import time


# Available locations
class Location(Enum):
    OSLO = 'OSL'
    TRONDHEIM = 'TRD'


class SamplingStrategy(Enum):
    OVERSAMPLING = 1
    UNDERSAMPLING = 2
    NONE = 3


LOCATION = Location.OSLO

# Only set to True when a final reproducible result is wanted
USE_FIXED_RANDOMSEED = True
SEED = 64 if USE_FIXED_RANDOMSEED else int(time.time())

TARGET_LABELS = ['On Time', 'Behind']

SVM_MODEL_PATH = '../../models/svm'
KNN_MODEL_PATH = '../../models/knn'
MLP_MODEL_PATH = '../../models/mlp'
XGBOOST_MODEL_PATH = '../../models/xgboost'

SVM_VISUALIZATION_PATH = '../../visualizations/models/svm'
KNN_VISUALIZATION_PATH = '../../visualizations/models/knn'
MLP_VISUALIZATION_PATH = '../../visualizations/models/mlp'
XGBOOST_VISUALIZATION_PATH = '../../visualizations/models/xgboost'

NUM_FOLDS = 5  # K-value for Cross-Validation


def setup():
    """Setup function to run before program start."""

    # Set up folder structure
    for directory in [
        SVM_MODEL_PATH, KNN_MODEL_PATH, MLP_MODEL_PATH, XGBOOST_MODEL_PATH,
        SVM_VISUALIZATION_PATH, KNN_VISUALIZATION_PATH, MLP_VISUALIZATION_PATH, XGBOOST_VISUALIZATION_PATH
    ]:
        os.makedirs(directory, exist_ok=True)
