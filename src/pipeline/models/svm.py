from sklearn.svm import SVC

from search import (
    grid_param_search,
    pipeline_search_params
)
from helpers import (
    create_pipeline,
    SamplingStrategy,
    evaluate_model,
    confusion_matrix,
    plot_learning_curve,
    timing,
    save_model
)
from config import (
    SVM_VISUALIZATION_PATH,
    SVM_MODEL_PATH
)


def svm(dl):
    X, y = dl.extract(50000)

    # search(X, y) <= this step is done

    # Best params from grid search
    model = create_pipeline(
        SVC(
            C=1.6,
            class_weight='balanced',
            gamma=0.00007,
            shrinking=True
        ),
        sampling_strategy=SamplingStrategy.UNDERSAMPLING,
        y=y
    )

    evaluate(X, y, model)
    # training_curve(X, y, model)
    # save_model(model, SVM_MODEL_PATH)


def search(X, y):
    svm_pipeline = create_pipeline(
        SVC(), sampling_strategy=SamplingStrategy.UNDERSAMPLING, y=y)

    # grid_param_search(X, y, svm_pipeline, pipeline_search_params('svc', {
    #    'C': [1, 2, 5, 10],
    #    'gamma': [0.01, 0.001, 0.0001],
    #    'class_weight': ['balanced', None],
    #    'shrinking': [True, False],
    # }))
    # 1st search used 50k samples and undersampling.
    # Best parameters: C=2, class_weight=balanced, gamma=0.0001, shrinking=true => score 0.47

    # Next => homing in on numerical values. Discard enumerated params (shringing + class_weight).
    # grid_param_search(X, y, svm_pipeline, pipeline_search_params('svc', {
    #    'C': [1.6, 1.85, 2, 2.15, 2.3],
    #    'gamma': [0.0001, 0.00012, 0.00008],
    #    'class_weight': ['balanced'],
    #    'shrinking': [True],
    # }))
    # 2nd search used 50k samples and undersampling.
    # Best parameters: C=1.6, class_weight=balanced, gamma=0.00008, shrinking=true => score 0.493

    # Next => homing in on numerical values. Discard enumerated params (shringing + class_weight).
    grid_param_search(X, y, svm_pipeline, pipeline_search_params('svc', {
        'C': [1.55, 1.6, 1.65],
        'gamma': [0.00007, 0.00008, 0.00009],
        'class_weight': ['balanced'],
        'shrinking': [True],
    }))
    # 3rd search used 50k samples and undersampling.
    # Best parameters: C=1.6, class_weight=balanced, gamma=0.00007, shrinking=true => score 0.494


def evaluate(X, y, model):
    with timing():
        scores, averages = evaluate_model(X, y, model)
        print(scores)
        print(f'Averages: {averages}')
        confusion_matrix(X, y, model, SVM_VISUALIZATION_PATH)


def training_curve(X, y, model):
    with timing():
        plot_learning_curve(X, y, model, SVM_VISUALIZATION_PATH)
