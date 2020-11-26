import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.construct import rand, random
from sklearn.preprocessing import LabelEncoder

from config import SEED


class DataLoader:
    """A class used to load training data

    Args:
        filePath (str): A relative path from main to the data file.

    Attributes:
        X (pandas.DataFrame): A dataframe of features
        y (pandas.DataFrame): A dataframe of targets

    """

    def __init__(self, filePath: str):
        print('[DataLoader] Loading data')
        df = pd.read_csv(filePath)

        # Set y as the target and select features
        y = df['TARGET']
        X = df[[
            'MONTH',
            'DAY_OF_MONTH',
            'WEEKDAY',
            'HOUR',
            'WIND_SPEED',
            'WIND_DIRECTION',
            'AIR_TEMPERATURE',
            'PRECIPITATION',
            'VISIBILITY'
        ]]

        # Parse string features to numeric values
        # using a labelencoder and inject them as features
        le = LabelEncoder()
        le.fit(df['AIRLINE_IATA'].values)
        X.insert(2, 'AIRLINE_IATA', le.transform(
            df['AIRLINE_IATA'].values), True)

        le = LabelEncoder()
        le.fit(df['GATE_STAND'].astype(str).values)
        X.insert(2, 'GATE_STAND', le.transform(
            df['GATE_STAND'].astype(str).values), True)

        le = LabelEncoder()
        le.fit(df['INT_DOM_SCHENGEN'].values)
        X.insert(2, 'INT_DOM_SCHENGEN', le.transform(
            df['INT_DOM_SCHENGEN'].values), True)

        le = LabelEncoder()
        le.fit(df['DEP_ARR'].values)
        X.insert(2, 'DEP_ARR', le.transform(df['DEP_ARR'].values), True)

        le = LabelEncoder()
        le.fit(df['TO_FROM'].values)
        X.insert(2, 'TO_FROM', le.transform(df['TO_FROM'].values), True)

        le = LabelEncoder()
        le.fit(df['FLIGHT_ID'].values)
        X.insert(2, 'FLIGHT_ID', le.transform(df['FLIGHT_ID'].values), True)

        self.X = X
        self.y = y

        del df

    def extract(self, count: int = None):
        """Extracts the data seperated as features and targets

        Args:
            count (int): The number of random samples to be returned. count=None returns all.

        Returns:
            bool: self.X and self.y as numpy arrays where X[i] belongs to y[i]

        Examples:
            To extract 20 random samples:
            >>> X, y = dataloader.extract(200)

            To extract all samples:
            >>> X, y = dataloader.extract()

        """

        X, y = self.X, self.y

        if not count == None:
            X = X.sample(count, random_state=SEED)
            y = y.sample(count, random_state=SEED)

        return X.values, y.values

    def histogram(self):
        """Shows a histogram plot of loaded features"""

        self.X.hist()
        plt.show()
