from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


from helpers import (
    create_pipeline,
    SamplingStrategy,
    evaluate_model,
    confusion_matrix,
    plot_learning_curve_keras,
    timing,
    generate_model_name
)
from config import (
    MLP_VISUALIZATION_PATH,
    MLP_MODEL_PATH,
    SEED
)

num_epochs = 200
batch_size = 4096
l2_value = 0.00001


def mlp(dl):
    X, y = dl.extract()

    model = create_model()
    print(model.summary())

    evaluate(X, y)
    # training_curve(X, y, model)
    # save_model(model)


def create_model():
    num_features = 15

    model = Sequential()
    model.add(BatchNormalization(input_dim=num_features))
    model.add(Dense(num_features, activation='relu',
                    kernel_initializer='he_uniform', kernel_regularizer=l2(l2_value)))
    model.add(Dense(256, activation='relu',
                    kernel_initializer='he_uniform', kernel_regularizer=l2(l2_value)))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu',
                    kernel_initializer='he_uniform', kernel_regularizer=l2(l2_value)))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu',
                    kernel_initializer='he_uniform', kernel_regularizer=l2(l2_value)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def evaluate(X, y):
    with timing():
        model_for_evaluation = KerasClassifier(
            build_fn=create_model, epochs=num_epochs, batch_size=batch_size, verbose=0)
        model = create_pipeline(
            model_for_evaluation, sampling_strategy=SamplingStrategy.OVERSAMPLING, y=y)
        scores, averages = evaluate_model(X, y, model, gpu_mode=True)
        print('\n\n', scores)
        print(f'Averages: {averages}')

        confusion_matrix(X, y, model, MLP_VISUALIZATION_PATH)


def training_curve(X, y, model):
    def mlp_databalancing(_X, _y, sampling_strategy):
        if sampling_strategy == SamplingStrategy.UNDERSAMPLING:
            _X, _y = RandomUnderSampler(random_state=SEED).fit_resample(_X, _y)
        elif sampling_strategy == SamplingStrategy.OVERSAMPLING:
            _X, _y = SMOTE(random_state=SEED, n_jobs=-1).fit_resample(_X, _y)
        return _X, _y

    SPLIT_AND_VALIDATE = True  # If False, train with all data
    SAMPLING_STRATEGY = SamplingStrategy.OVERSAMPLING

    with timing():
        # Custom code for databalancing for MLP, since imblearn pipeline don't return keras-history
        if SPLIT_AND_VALIDATE:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.1, random_state=SEED)

            if SAMPLING_STRATEGY != SamplingStrategy.NONE:
                X_train, y_train = mlp_databalancing(
                    X_train, y_train, SAMPLING_STRATEGY)

            history = model.fit(X_train, y_train, epochs=num_epochs,
                                batch_size=batch_size, validation_data=(X_val, y_val))
        else:
            X, y = mlp_databalancing(X, y, SAMPLING_STRATEGY)
            history = model.fit(X, y, epochs=num_epochs,
                                batch_size=batch_size, verbose=0)

        plot_learning_curve_keras(history)


def save_model(model):
    model_name = generate_model_name()
    model.save(f'{MLP_MODEL_PATH}/{model_name}')
