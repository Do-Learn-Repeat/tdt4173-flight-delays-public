import pandas as pd
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
import math
import seaborn as sns

# TODO: These should ideally be imported from a python script...
CLEAN_DATA_PATH = "../../data/clean"
FLIGHTS_PROCESSED_PATH = "../../data/processed/flights"

VISUALIZATIONS_PATH = "../../data/visualizations"

# TODO The TARGET values should be imported from somewhere
"""
0 = cancelled
1 = ahead OR on time
2 = behind
"""
NUMBER_OF_TARGETS = 3

TARGET_CLASSES = ["Cancelled", "Ahead or on time", "Behind"]

# These are reset at every location loop
numeric_columns = []
non_numeric_columns = []


def main():
    for root, dirs, files in os.walk(CLEAN_DATA_PATH):
        for file in files:
            process(file)


def process(file):
    df = load_clean_data(file)

    location = file.split('.')[0]

    location_directory = os.path.join(VISUALIZATIONS_PATH, location)
    # Create output directory
    if not os.path.exists(location_directory):
        os.makedirs(location_directory)

    # Sort into numeric and non-numeric columns
    numeric_columns.clear()
    non_numeric_columns.clear()
    for column_name, _ in df.iteritems():
        if (pd.to_numeric(df[column_name], errors='coerce').notnull().all()):
            numeric_columns.append(column_name)
        else:
            non_numeric_columns.append(column_name)

    target_distribution(df, location)
    target_per_hour(df, location)
    target_per_month(df, location)
    histogram_per_column(df, location)
    range_per_column(df, location)
    violin_plot(df, location)
    histogram_of_delays(location)


def save_plt_figure(fig, location, filename):
    plt.savefig(os.path.join(VISUALIZATIONS_PATH,
                             location, "{}.png".format(filename)))


def load_clean_data(file):
    """
    Loads the clean data CSV for a specific location and returns it as a dataframe.
    """
    print("Loading clean data for", file.split('.')[0])

    df = pd.read_csv(os.path.join(CLEAN_DATA_PATH, file))
    return df


def histogram_of_delays(location):
    df = pd.DataFrame()

    # Load all .csv files from the processed location folder
    location_path = os.path.join(FLIGHTS_PROCESSED_PATH, location)
    for root, dirs, files in os.walk(location_path):
        for file in files:
            if "sample" in file:
                continue
            data = pd.read_csv(os.path.join(location_path, file))
            df = pd.concat([df, data["DELAY"]], ignore_index=True)

    delays = df[0].values
    delays = delays[np.isfinite(delays)]  # Remove inf values
    fig = plt.figure(figsize=(10, 10))

    # With range
    plt.subplot(1, 2, 1)
    # Most flights have this delay interval
    plt.hist(delays, bins=50, range=(-50, 100))

    # Without range
    plt.subplot(1, 2, 2)
    plt.hist(delays, bins=50)

    save_plt_figure(fig, location, "delays")
    plt.show()
    plt.close()


def histogram_per_column(df, location):
    bins = {
        "TARGET": 3,
        "YEAR": 4
    }
    fig = plt.figure(figsize=(20, 20))
    plt_dimension = math.ceil(math.sqrt(len(numeric_columns)))
    index_in_suplot = 0
    for column_name in numeric_columns:
        index_in_suplot += 1
        plt.subplot(plt_dimension, plt_dimension, index_in_suplot)
        if column_name in bins:
            print(bins[column_name])
            plt.hist(df[column_name].values, bins[column_name])
        else:
            plt.hist(df[column_name].values)
        plt.title(column_name)
    save_plt_figure(fig, location, "hist_per_column")
    plt.show()
    plt.close()


def range_per_column(df, location):
    fig = plt.figure(figsize=(20, 20))
    plt_dimension = math.ceil(math.sqrt(len(numeric_columns)))
    index_in_suplot = 0
    for column_name in numeric_columns:
        index_in_suplot += 1
        plt.subplot(plt_dimension, plt_dimension, index_in_suplot)
        plt.boxplot(df[column_name].values)
        plt.title(column_name)
    save_plt_figure(fig, location, "range_per_column")
    plt.show()
    plt.close()


def violin_plot(df, location):
    fig = plt.figure(figsize=(20, 20))
    plt_dimension = math.ceil(math.sqrt(len(numeric_columns)))
    index_in_suplot = 0
    for column_name in numeric_columns:
        index_in_suplot += 1
        plt.subplot(plt_dimension, plt_dimension, index_in_suplot)
        sns.violinplot(data=df[column_name].values)
        plt.title(column_name)
    save_plt_figure(fig, location, "violin_plot_per_column")
    plt.show()
    plt.close()


def target_distribution(df, location):
    fig = plt.figure(figsize=(8, 8))
    distribution = []
    for i in range(NUMBER_OF_TARGETS):
        subset_df = df[df["TARGET"] == i]
        subset_df = subset_df["TARGET"]
        distribution.append(subset_df.count())

    plt.bar(TARGET_CLASSES, distribution)
    plt.title(location)
    plt.xlabel("Target")
    plt.ylabel("Count")
    save_plt_figure(fig, location, "target_distribution")
    # plt.show()
    plt.close()


def target_per_column(df, column_name, filename_prefix, location):
    fig = plt.figure(figsize=(8, 8))
    for i in range(NUMBER_OF_TARGETS):
        subset_df = df[df["TARGET"] == i]
        subset_df = subset_df[[column_name, "TARGET"]]
        values = subset_df.groupby(subset_df[column_name]).size()
        plt.plot(values)

    plt.title(location)
    plt.legend(TARGET_CLASSES)
    plt.xlabel(column_name)
    plt.ylabel("Count")
    save_plt_figure(fig, location, "{}_per_target".format(filename_prefix))
    # plt.show()
    plt.close()


def target_per_hour(df, location):
    target_per_column(df, "HOUR", "hourly", location)


def target_per_month(df, location):
    target_per_column(df, "MONTH", "monthly", location)


if __name__ == "__main__":
    main()
