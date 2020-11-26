from config import setup, LOCATION
from dataloader import DataLoader

from models.baseline import baseline
from models.svm import svm
from models.knn import knn
from models.mlp import mlp
from models.xgboost import xgboost


def main():
    # Run the setup to create folders etc
    setup()

    dl = DataLoader(f'../../data/clean/{LOCATION.value}.csv')
    # dl.histogram()

    # Method experimentation is done inside each model module:
    baseline(dl)
    # svm(dl)
    # knn(dl)
    # xgboost(dl)
    # mlp(dl)


if __name__ == '__main__':
    main()
