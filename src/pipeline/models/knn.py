from sklearn.neighbors import KNeighborsClassifier

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
    KNN_VISUALIZATION_PATH,
    KNN_MODEL_PATH
)


def knn(dl):
    X, y = dl.extract()

    # search(X, y) # <= this step is done

    # Best params from grid search
    model = create_pipeline(
        KNeighborsClassifier(n_neighbors=10, weights='uniform'),
        sampling_strategy=SamplingStrategy.UNDERSAMPLING, y=y
    )

    evaluate(X, y, model)
    # training_curve(X, y, model)
    # save_model(model, KNN_MODEL_PATH)


def search(X, y):
    knn_pipeline = create_pipeline(KNeighborsClassifier(
        1), sampling_strategy=SamplingStrategy.UNDERSAMPLING, y=y)

    # gridParamSearch(X, y, knn_pipeline, pipeline_search_params('kneighborsclassifier', {
    #    'n_neighbors': [1, 10, 50, 100],
    #    'weights': ['distance', 'uniform'],
    # }))
    # 1st search used 500k samples and undersampling.
    # Best parameters: K = 10, weights = 'uniform'

    # Next step: search the area around k=10
    # gridParamSearch(X, y, knn_pipeline, pipeline_search_params('kneighborsclassifier', {
    #    'n_neighbors': [6, 8, 10, 12, 14],
    #    'weights': ['distance', 'uniform'],
    # }))
    # 2nd search used 500k samples and undersampling.
    # Scored 0.5437. Best parameters: K = 6, weights = 'uniform'

    # Next step: search the area around k=6
    grid_param_search(X, y, knn_pipeline, pipeline_search_params('kneighborsclassifier', {
        'n_neighbors': [5, 6, 7],
        'weights': ['distance', 'uniform'],
    }))
    # 3rd search used 500k samples and undersampling.
    # Scored 0.5437. Best parameters: K = 6, weights = 'uniform'


def evaluate(X, y, model):
    with timing():
        scores, averages = evaluate_model(X, y, model)
        print(scores)
        print(f'Averages: {averages}')

        confusion_matrix(X, y, model, KNN_VISUALIZATION_PATH)


def training_curve(X, y, model):
    with timing():
        plot_learning_curve(X, y, model, KNN_VISUALIZATION_PATH)
