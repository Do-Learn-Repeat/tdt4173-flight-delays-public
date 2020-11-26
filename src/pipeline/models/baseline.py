import random
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from collections import Counter

from config import SEED


def baseline(dl):
    X, y = dl.extract()

    y_test = train_test_split(X, y, test_size=.2, random_state=SEED)[3]

    targets = Counter(y_test)

    micros = []
    macros = []

    # We average the baseline over 10 runs, to get a more stable score
    for _ in range(10):
        y_baseline_pred = [weighted_dice(targets) for _ in y_test]
        f1_micro = f1_score(y_test, y_baseline_pred, average='micro')
        f1_macro = f1_score(y_test, y_baseline_pred, average='macro')
        micros.append(f1_micro)
        macros.append(f1_macro)

    micros = np.array(micros)
    macros = np.array(macros)
    print(
        f'Baseline Average scores => f1_macro: {macros.mean():.4f} \tf1_micro: {micros.mean():.4f}')


def weighted_dice(targets):
    '''
    Takes a dict of classification targets and their frequencies in the dataset.
    The keys are the target-values.
    '''
    return random.choices(list(targets.keys()), weights=targets.values(), k=1)[0]
