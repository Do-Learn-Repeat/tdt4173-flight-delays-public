from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time


def grid_param_search(X, y, model, param_dist):
    """Prints the best parameters for the model using grid search.

    Args:
        X (pandas DataFrame): A dataframe containing features.
        y (pandas DataFrame): A dataframe containing targets.
        model (sklearn Model): The model for which search should apply.
        param_dist (dict): A dict where {[parameter_name]: parameter values}.

    Example:
        >>> model = create_model()
        >>> X, y = get_data()
        >>> param_dist = {'param1': [1, 2], 'param2': ['uniform', 'distance']}
        >>> grid_param_search(X, y, model, param_dist)

    """

    print('[GRID SEARCH (CV)] Finding best model using parameters:')
    print(param_dist, end='\n\n')

    # Perform a grid search on the available parameters
    search = GridSearchCV(model, param_grid=param_dist,
                          n_jobs=-1, scoring='f1_macro')
    start_time = time.time()

    # perform search
    search.fit(X, y)

    # Print stats and solution
    time_diff = time.time() - start_time
    num_candidates = len(search.cv_results_['params'])
    print(
        f'[GRID SEARCH (CV)] Used {time_diff:.2f} seconds for {num_candidates} candidate parameter settings.')
    print('\nBest candidate:')
    print(f'Params: {search.best_params_}')
    print(f'Score: {search.best_score_}')


def random_param_search(X, y, model, num_searches, param_dist):
    """Prints the best parameters for the model using random search.

    Args:
        X (pandas DataFrame): A dataframe containing features.
        y (pandas DataFrame): A dataframe containing targets.
        model (sklearn Model): The model for which search should apply.
        num_searches (int): How many random searches should be performed.
        param_dist (dict): A dict where {[parameter_name]: parameter values}.

    Example:
        >>> model = create_model()
        >>> X, y = get_data()
        >>> param_dist = {'param1': [1, 2, 3, 4, 5], 'param2': ['uniform', 'distance']}
        >>> random_param_search(X, y, model, 3, param_dist)

    """

    print('[RANDOM SEARCH (CV)] Finding best model using parameters:')
    print(param_dist, end='\n\n')

    # Perform a {num_searches} random searchs on a random selection of the
    # provided parameters.
    search = RandomizedSearchCV(model, param_distributions=param_dist,
                                n_jobs=-1, scoring='f1_macro', n_iter=num_searches)
    start_time = time.time()
    search.fit(X, y)

    # Print stats and solution
    time_diff = time.time() - start_time
    print(
        f'[RANDOM SEARCH (CV)] Used {time_diff:.2f} seconds for {num_searches} candidate parameter settings.')
    print('\nBest candidate:')
    print(f'Params: {search.best_params_}')
    print(f'Score: {search.best_score_}')


def pipeline_search_params(model_name, params):
    """Wraps a param_dict inside the model name for wrapped models (pipelines).

    Args:
        model_name (str): The name of the model (model type)
        params (dict): The param dict used for a search function.

    Returns:
        dict: A dict where every value is wrapped in the model name.

    """

    # Synthetic sugar for models wrapped in pipelines
    return {model_name+'__' + key: params[key] for key in params}
