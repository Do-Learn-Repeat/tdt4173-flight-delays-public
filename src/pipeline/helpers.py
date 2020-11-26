import pandas as pd
import numpy as np
import seaborn
import time
from datetime import datetime
import matplotlib.pyplot as plt
import time
import pickle

from config import SamplingStrategy, SEED, LOCATION, NUM_FOLDS, TARGET_LABELS

# For evaluation
from sklearn.metrics import confusion_matrix as cm
from yellowbrick.model_selection import LearningCurve
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from collections import Counter


def create_pipeline(model, sampling_strategy, y):
    """Wraps a model in a pipeline to resample training data.

    Args:
        model (sklearn Model): The model to wrap.
        sampling_strategy (SamplingStrategy): The sampling strategy for the pipeline.
        y (pandas Dataframe): A dataframe containing targets.

    Returns:
        sklearn pipeline: A pipeline wrapping the model.

    """

    balancer = 'passthrough'

    if sampling_strategy == SamplingStrategy.UNDERSAMPLING:
        # We want to use a random undersampler if we use
        # undersampling is the resample strategy
        databalancing_stats(y, sampling_strategy)
        balancer = RandomUnderSampler(random_state=SEED)

    elif sampling_strategy == SamplingStrategy.OVERSAMPLING:
        # We want to use a SMOTE, the most common oversampler,
        # if we use oversampling is the resample strategy
        databalancing_stats(y, sampling_strategy)
        balancer = SMOTE(random_state=SEED, n_jobs=-1)

    return make_pipeline(balancer, model)


def databalancing_stats(y, sampling_strategy):
    """Prints a visualization of pre vs post resampling target distribution.

    (Currently only works for binary problems)

    Args:
        _y (numpy array): training data targets.
        sampling_strategy (SamplingStrategy): The sampling strategy to visualize.

    """

    original_count = Counter(y)
    print('Following statistics is to get an idea about what will be done to the training-set(s):')
    print(f'Targets before databalancing: {original_count}')
    if sampling_strategy == SamplingStrategy.OVERSAMPLING:
        max_key = max(original_count, key=original_count.get)
        new_count = f'0: {original_count[max_key]} 1: {original_count[max_key]}'
    else:
        min_key = min(original_count, key=original_count.get)
        new_count = f'0: {original_count[min_key]} 1: {original_count[min_key]}'
    print(f'Targets after databalancing: {new_count} \n')


def evaluate_model(X, y, model, gpu_mode=False):
    """Evaulates the model using k-fold kross validation.

    Args:
        X (numpy array): Features for the evaluation.
        y (numpy array): Targets for the evaluation.
        model (sklearn Model): The model for which search should apply.
        gpu_mode (bool): Uses 1 worker to avoid crashing gpu systems if True.

    Returns:
        (dict, dict): where [0] is scores over all runs and [1] is average scores.

    """

    # create a stratified cross validation to ensure good sample
    # representation.
    cv = StratifiedKFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)
    N_JOBS = 1 if gpu_mode else -1

    scores = cross_validate(model, X, y, scoring=[
                            'f1_macro', 'f1_micro'], cv=cv, n_jobs=N_JOBS)

    # Iterate over all folds and find the average scores
    averages = {}
    for key, val in scores.items():
        averages[key] = val.mean()

    return pd.DataFrame(scores), averages


def confusion_matrix(X, y, model, figPath=None):
    """Prints a confusion matrix for a model (with file saving ability).

    Args:
        X (numpy array): Features for the evaluation.
        y (numpy array): Targets for the evaluation.
        model (sklearn Model): The model to visualize.
        figPath (str): Where to save the figure. figPath=None does not save the figure.

    """

    # Split the data into 80% - 20%, where the majority is used
    # to train the model and the rest is used for testing.
    # The test data is used to plot the matrix.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)
    model = model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    data = cm(y_test, predicted)

    # Below is from https://onestopdataanalysis.com/confusion-matrix-python/
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))

    plt.title('Confusion Matrix')

    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(data, annot=True, fmt='g',
                         cmap='YlGnBu', cbar_kws={'label': 'Scale'})

    ax.set_xticklabels((TARGET_LABELS[0], TARGET_LABELS[1]))
    ax.set_yticklabels((TARGET_LABELS[0], TARGET_LABELS[1]))

    ax.set(ylabel='True Label', xlabel='Predicted Label')

    if not figPath == None:
        plt.savefig(f'{figPath}/confusion.png')

    plt.show()


def plot_learning_curve(X, y, model, figPath=None):
    """Prints a training curve for a model (with file saving ability).

    Args:
        X (numpy array): Features for the evaluation.
        y (numpy array): Targets for the evaluation.
        model (sklearn Model): The model to visualize.
        figPath (str): Where to save the figure. figPath=None does not save the figure.

    """

    # create a stratified cross validation to ensure good sample
    # representation.
    cv = StratifiedKFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)

    visualizer = LearningCurve(
        model, cv=cv, scoring='f1_weighted', train_sizes=np.linspace(0.3, 1.0, 10), n_jobs=-1)
    visualizer.fit(X, y)

    if figPath == None:
        visualizer.show()
    else:
        visualizer.show(outpath=f'{figPath}/learning.png')


def plot_learning_curve_keras(history):
    """Prints a training curve for a Keras model.

    Args:
        history (keras history object): History object obtained from fitting.

    """

    # Plotting Keras learning curve
    # From: https://stackabuse.com/python-for-nlp-multi-label-text-classification-with-keras/

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def generate_model_name():
    """Generates a model name from the current time."""

    now = datetime.now()
    return f"{LOCATION.value}-{now.strftime('%d-%m-%Y-%H%M%S')}"


def save_model(model, path):
    """Saves a model using Pickle. (sklearn only)

    Args:
        model (sklearn model): The model to save.
        path (str): Path to the file location.

    """

    model_name = generate_model_name()
    pickle.dump(model, open(f'{path}/{model_name}', 'wb'))


class timing:
    """With-statement to time a code block.

    Example:
        >>> with timing():
        >>>     expensive_operation()
        Finished in 1.32 seconds

    """

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, exc, tb):
        timeDiff = time.time() - self.start_time
        print(f'Finished in {timeDiff:.3f} seconds')
