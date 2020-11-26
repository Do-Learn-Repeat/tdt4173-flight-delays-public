from xgboost import XGBClassifier

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
    XGBOOST_VISUALIZATION_PATH,
    XGBOOST_MODEL_PATH
)


def xgboost(dl):
    X, y = dl.extract(200000)

    # search(X, y, SEED) # <= this step is done

    # Best params from search
    model = create_pipeline(
        XGBClassifier(
            gamma=8,
            learning_rate=0.08,
            max_depth=22,
        ),
        sampling_strategy=SamplingStrategy.UNDERSAMPLING,
        y=y
    )

    evaluate(X, y, model)
    # training_curve(X, y, model)
    # save_model(model, XGBOOST_MODEL_PATH)


def search(X, y):
    xgb_pipeline = create_pipeline(
        XGBClassifier(), sampling_strategy=SamplingStrategy.UNDERSAMPLING, y=y)

    # gridParamSearch(X, y, xgb_pipeline, pipeline_search_params('xgbclassifier', {
    #    'gamma': [0, 5, 10, 20],
    #    'learning_rate': [0.1, 0.2, 0.3],
    #    'max_depth': [5, 10, 20, 30],
    # }))
    # 1st search used 250k samples and undersampling.
    # Best parameters: gamma=10, learning_rate=0.1, max_depth=30 => score: 0.6114179052808012

    # Next: Focus parameters
    grid_param_search(X, y, xgb_pipeline, pipeline_search_params('xgbclassifier', {
        'gamma': [8, 10, 12, 14],
        'learning_rate': [0.08, 0.1, 0.12],
        'max_depth': [5, 10, 20, 30],
    }))
    # 1st search used 250k samples and undersampling (locking learning rate).
    # Best parameters: gamma=8, learning_rate=0.08, max_depth=20 => score: 0.6114179052808012


def evaluate(X, y, model):
    with timing():
        scores, averages = evaluate_model(X, y, model)
        print(scores)
        print(f'Averages: {averages}')

        confusion_matrix(X, y, model, XGBOOST_VISUALIZATION_PATH)


def training_curve(X, y, model):
    with timing():
        plot_learning_curve(X, y, model, XGBOOST_VISUALIZATION_PATH)
