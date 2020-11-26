import pandas as pd
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
import math
import seaborn as sns
from collections import Counter

# TODO: These should ideally be imported from a python script...
CLEAN_DATA_PATH = "../../data/clean"
FLIGHTS_PROCESSED_PATH = "../../data/processed/flights"

VISUALIZATIONS_PATH = "../../visualizations/data/"

columns_to_ignore = ["FLIGHT_ID", "GATE_STAND",
                     "AIRLINE_IATA", "TO_FROM", "DEP_ARR"]

# TODO These column names should be imported from somewhere
VISIBILITY = "VISIBILITY"
TARGET = "TARGET"
DELAY = "DELAY"
INT_DOM_SCHENGEN = "INT_DOM_SCHENGEN"
PRECIPITATION = "PRECIPITATION"
YEAR = "YEAR"
MONTH = "MONTH"
DAY_OF_MONTH = "DAY_OF_MONTH"
WEEKDAY = "WEEKDAY"
HOUR = "HOUR"
WIND_SPEED = "WIND_SPEED"
WIND_DIRECTION = "WIND_DIRECTION"
AIR_TEMPERATURE = "AIR_TEMPERATURE"

"""
0 = On time
1 = Delayed
"""
TARGET_CLASSES = ["On time", "Delayed"]

"""
0 = cancelled
1 = ahead OR on time
2 = behind
"""
TARGET_CLASSES_THREE = ["Cancelled", "Ahead or on time", "Behind"]

# These are reset at every location loop
numeric_columns = []
non_numeric_columns = []

WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday', 'Sunday']


INT_DOM_SCHENGEN_CLASSES = ["I", "D", "S"]


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
        if column_name in columns_to_ignore:
            continue
        if (pd.to_numeric(df[column_name], errors='coerce').notnull().all()):
            numeric_columns.append(column_name)
        else:
            non_numeric_columns.append(column_name)

    target_distribution(df, location)
    target_per_hour(df, location)
    target_per_month(df, location)
    plot_per_column(df, location)
    range_per_column(df, location)
    violin_plot(df, location)
    histogram_of_delays(location)
    delays_on_int_dom_schengen(location)
    average_delays_on_int_dom_schengen(location)  # Not used
    barplot(df, location)
    average_delay_per_weekday(df, location)
    visibility_vs_delay_hist(df, location)
    average_short_visibility_vs_delay(df, location)
    range_delay_per_weekday(df, location)
    range_delay_per_int_dom_sch(df, location)


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


def sort_dict_by_key(d):
    r = {}
    for key in sorted(d.keys()):
        r[key] = d[key]
    return r


def visibility_vs_delay_hist(df, location):
    fig = plt.figure(figsize=(12, 6))
    visibilities = df[VISIBILITY].to_numpy()
    delays = df[DELAY].to_numpy()

    bins = (15, 15)
    plt_rows = 1
    plt_columns = 2
    xlabel = "Visibility in meters"
    ylabel = "Delay in minutes"

    plt.subplot(plt_rows, plt_columns, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.hist2d(visibilities, delays, bins=bins)

    plt.subplot(plt_rows, plt_columns, 2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.hist2d(visibilities, delays, bins=bins, range=(
        (min(visibilities), max(visibilities)), (-40, 40)))

    save_plt_figure(fig, location, "visibility_vs_delay_hist")
    # plt.show()
    plt.close()


def range_delay_per_weekday(df, location):
    x = []
    for i in range(len(WEEKDAYS)):
        weekday_df = df.loc[df["WEEKDAY"] == i]
        y = weekday_df[DELAY].to_numpy()
        x.append(y)

    fig = plt.figure(figsize=(10, 6))
    plt.boxplot(x, labels=WEEKDAYS)
    plt.xlabel("Weekday")
    plt.ylabel("Delay in minutes")

    save_plt_figure(fig, location, "delays-on-weekdays-range")
    # plt.show()
    plt.close()


def range_delay_per_int_dom_sch(df, location):
    x = []
    for s in INT_DOM_SCHENGEN_CLASSES:
        temp_df = df.loc[df[INT_DOM_SCHENGEN] == s]
        y = temp_df[DELAY].to_numpy()
        x.append(y)

    fig = plt.figure(figsize=(8, 2))
    plt.boxplot(x, labels=["International",
                           "Domestic", "Schengen"], vert=False)
    plt.ylabel("Region")
    plt.xlabel("Delay in minutes")

    plt.tight_layout()

    save_plt_figure(fig, location, "boxplot_int_dom_schengen_delay")
    # plt.show()
    plt.close()

    fig = plt.figure(figsize=(8, 2))
    plt.boxplot(x, labels=["International", "Domestic",
                           "Schengen"], showfliers=False, vert=False)
    plt.ylabel("Region")
    plt.xlabel("Delay in minutes")

    plt.tight_layout()

    save_plt_figure(
        fig, location, "boxplot_int_dom_schengen_delay_without_outliers")
    # plt.show()
    plt.close()


def average_short_visibility_vs_delay(df, location):
    averages = []
    visibility_intervals = [0, 500, 1000, 1500, 2000, 3000, 4000, 6000]
    labels = []
    for i in range(1, len(visibility_intervals)):
        temp_df = df.loc[(df[VISIBILITY] >= visibility_intervals[i - 1])
                         & (df[VISIBILITY] < visibility_intervals[i])]
        average = temp_df[DELAY].mean(axis=0)
        averages.append(average)

        label = "{}-{}".format(visibility_intervals[i - 1],
                               visibility_intervals[i])
        labels.append(label)

    # Add last
    temp_df = df.loc[df[VISIBILITY] > visibility_intervals[-1]]
    labels.append(
        "{}-{}".format(visibility_intervals[-1], max(temp_df[VISIBILITY])))
    average = temp_df[DELAY].mean(axis=0)
    averages.append(average)

    fig = plt.figure(figsize=(10, 3))
    plt.bar(labels, averages)
    plt.xlabel("Visibility in meters")
    plt.ylabel("Average delay in minutes")
    plt.tight_layout()

    save_plt_figure(fig, location, "average_visibility_vs_delay")
    # plt.show()
    plt.close()


def average_delay_per_weekday(df, location):
    averages = []
    for i in range(len(WEEKDAYS)):
        weekday_df = df.loc[df["WEEKDAY"] == i]
        average = weekday_df[DELAY].mean(axis=0)
        averages.append(average)

    fig = plt.figure(figsize=(10, 6))
    plt.bar(WEEKDAYS, averages)
    plt.xlabel("Weekday")
    plt.ylabel("Average delay in minutes")

    save_plt_figure(fig, location, "average_delays_on_weekdays")
    # plt.show()
    plt.close()


# ! Not used
# def average_delays_on_int_dom_schengen(location):
#     df=pd.DataFrame()

#     # Load all .csv files from the processed location folder
#     location_path=os.path.join(FLIGHTS_PROCESSED_PATH, location)
#     for root, dirs, files in os.walk(location_path):
#         for file in files:
#             if "sample" in file:
#                 continue
#             data=pd.read_csv(os.path.join(location_path, file))
#             df=pd.concat(
#                 [df, data[[DELAY, INT_DOM_SCHENGEN]]], ignore_index = True)

#     # Remove inf values
#     delays=df[DELAY]
#     df[DELAY]=delays[np.isfinite(delays)]
#     df.dropna(subset = [DELAY], inplace = True)

#     averages=[]
#     labels=[]
#     step=25
#     for i in range(-50, 120, step):
#         temp_df=df.loc[(df[DELAY] >= i) & (df[DELAY] < i + step])]
#         average =temp_df[DELAY].mean(axis = 0)
#         averages.append(average)

#         label="{}-{}".format(i, i + step)
#         labels.append(label)

#     fig=plt.figure(figsize= (10, 3))
#     plt.bar(labels, averages)
#     plt.xlabel("Average delay ")
#     plt.ylabel("Average delay in minutes")
#     plt.tight_layout()

#     save_plt_figure(fig, location, "average_visibility_vs_delay")
#     # plt.show()
#     plt.close()


# def delays_per_weekday(df, location):
#     # df_counts = df.groupby("WEEKDAY")["WEEKDAY"].count().to_dict()
#     weekdays = ['Monday', 'Tuesday', 'Wednesday',
#                 'Thursday', 'Friday', 'Saturday', 'Sunday']
#     # fig = plt.figure(figsize=(10, 6))
#     # plt.bar(weekdays, df_counts.values())
#     # # plt.plot(weekdays, df_counts.values())
#     # plt.xlabel("Weekday")
#     # plt.ylabel("Number of flights")
#     # # plt.show()
#     # save_plt_figure(fig, location, "delays-on-weekdays")
#     # plt.close()
#     df = pd.DataFrame()

#     # Load all .csv files from the processed location folder
#     location_path = os.path.join(FLIGHTS_PROCESSED_PATH, location)
#     for root, dirs, files in os.walk(location_path):
#         for file in files:
#             if "sample" in file:
#                 continue
#             data = pd.read_csv(os.path.join(location_path, file))
#             df = pd.concat(
#                 [df, data[[DELAY, WEEKDAY]]], ignore_index=True)

#     delays = df[DELAY]
#     delays = delays[np.isfinite(delays)]  # Remove inf values
#     df[DELAY] = delays

#     delays = []
#     for i in range(len(weekdays)):
#         x = df.loc[df[WEEKDAY] == i]
#         delays.append(x[DELAY])
#         print(x[DELAY].count())

#     fig = plt.figure(figsize=(10, 6))
#     # colors = ["Blue", "Orange", "Green"]
#     plt.hist(delays, bins=30, range=(-50, 100), label=weekdays, histtype="bar")
#     plt.legend(loc="upper right")
#     plt.legend(prop={'size': 10})
#     plt.xlabel("Delay in minute")
#     plt.ylabel("Number of flights")

#     save_plt_figure(fig, location, "delays-on-weekdays")
#     # plt.show()
#     plt.close()


def delays_on_int_dom_schengen(location):
    df = pd.DataFrame()

    # Load all .csv files from the processed location folder
    location_path = os.path.join(FLIGHTS_PROCESSED_PATH, location)
    for root, dirs, files in os.walk(location_path):
        for file in files:
            if "sample" in file:
                continue
            data = pd.read_csv(os.path.join(location_path, file))
            df = pd.concat(
                [df, data[[DELAY, INT_DOM_SCHENGEN]]], ignore_index=True)

    delays = df[DELAY]
    delays = delays[np.isfinite(delays)]  # Remove inf values
    df[DELAY] = delays

    international = df.loc[df[INT_DOM_SCHENGEN] == "I"]
    domestic = df.loc[df[INT_DOM_SCHENGEN] == "D"]
    schengen = df.loc[df[INT_DOM_SCHENGEN] == "S"]

    delays = [international[DELAY], domestic[DELAY], schengen[DELAY]]

    fig = plt.figure(figsize=(10, 6))
    colors = ["Blue", "Orange", "Green"]
    labels = ["International", "Domestic", "Schengen"]
    plt.hist(delays, bins=30, range=(-50, 100),
             color=colors, label=labels, histtype="bar")
    plt.legend(loc="upper right")
    plt.legend(prop={'size': 10})
    plt.xlabel("Delay in minutes")
    plt.ylabel("Number of flights")

    save_plt_figure(fig, location, "delays-on-int-dom-schengen")
    # plt.show()
    plt.close()


def histogram_of_delays(location):
    df = pd.DataFrame()

    # Load all .csv files from the processed location folder
    location_path = os.path.join(FLIGHTS_PROCESSED_PATH, location)
    for root, dirs, files in os.walk(location_path):
        for file in files:
            if "sample" in file:
                continue
            data = pd.read_csv(os.path.join(location_path, file))
            df = pd.concat([df, data[DELAY]], ignore_index=True)

    delays = df[0].values
    delays = delays[np.isfinite(delays)]  # Remove inf values

    ### Double plot ###
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    # Most flights have this delay interval
    plt.hist(delays, bins=50, range=(-50, 100))
    plt.xlabel("Delay in minutes")
    plt.ylabel("Number of flights")

    # Without range
    plt.subplot(1, 2, 2)
    plt.hist(delays, bins=50)
    plt.xlabel("Delay in minutes")
    plt.ylabel("Number of flights")

    plt.tight_layout()

    save_plt_figure(fig, location, "delays")
    # plt.show()
    plt.close()

    ### Single plot ###
    fig = plt.figure(figsize=(10, 3))
    plt.hist(delays, bins=50, range=(-50, 100))
    plt.xlabel("Delay in minutes")
    plt.ylabel("Number of flights")
    plt.tight_layout()
    save_plt_figure(fig, location, "delays_cropped")
    # plt.show()
    plt.close()


def barplot(df, location):
    fig = plt.figure(figsize=(20, 28))
    plt_rows = 5
    plt_columns = 3
    index_in_suplot = 0
    for column_name in numeric_columns:
        index_in_suplot += 1
        plt.subplot(plt_rows, plt_columns, index_in_suplot)
        row_counter = Counter(df[column_name].values)
        x = np.arange(len(row_counter))
        plt.bar(x, row_counter.values())
        plt.xticks(x, row_counter.keys())
        plt.title(column_name)
    for column_name in non_numeric_columns:
        index_in_suplot += 1
        plt.subplot(plt_rows, plt_columns, index_in_suplot)
        row_counter = Counter(df[column_name].values)
        x = np.arange(len(row_counter))
        plt.bar(x, row_counter.values())
        plt.xticks(x, row_counter.keys())
        plt.title(column_name)
    save_plt_figure(fig, location, "barplot")
    # plt.show()
    plt.close()


def plot_per_column(df, location):
    graphs = {
        YEAR: 4,
        MONTH: 12,
        WEEKDAY: 7,
        DAY_OF_MONTH: 31,
    }

    # Columns that are not useful to put in histograms/violin plots
    bars = {
        INT_DOM_SCHENGEN: 3,
        "DEP_ARR": 2,
        TARGET: 3,
        # "AIRLINE_IATA": nan
        # "FLIGHT_ID": nan
        # "TO_FROM": nan
    }

    numeric_histogram_columns_with_bins = {
        WIND_DIRECTION: 36,  # 10 degree bins
        HOUR: 24,
        WIND_SPEED: 15,
    }

    lables = {
        YEAR: "Year",
        MONTH: "Month",
        DAY_OF_MONTH: "Day of month",
        WEEKDAY: "Weekday",
        HOUR: "Hour",
        PRECIPITATION: "Precipication in millimeters",
        WIND_DIRECTION: "Wind direction in degrees",
        VISIBILITY: "Visibility in meters",
        AIR_TEMPERATURE: "Air temperature in Celsius",
        WIND_SPEED: "Wind speed in m/s",
        TARGET: "On-time vs. delayed",
        DELAY: "Delay in minutes",
        INT_DOM_SCHENGEN: "Schengen vs. domestic vs. international"
    }

    # violin_plots = [VISIBILITY]
    violin_plots = []

    fig = plt.figure(figsize=(10, 14))
    plt_rows = 5
    plt_columns = 3
    index_in_suplot = 0
    for column_name in numeric_columns:
        index_in_suplot += 1
        plt.subplot(plt_rows, plt_columns, index_in_suplot)
        if column_name in graphs:
            row_counter = Counter(df[column_name].values)
            row_counter = sort_dict_by_key(row_counter)
            if(column_name == YEAR or column_name == MONTH):
                plt.xticks(np.arange(min(row_counter.keys()),
                                     max(row_counter.keys()) + 1, 1.0))
            plt.plot(row_counter.keys(), row_counter.values())
        elif column_name in bars:
            row_counter = Counter(df[column_name].values)
            row_counter = sort_dict_by_key(row_counter)
            x = np.arange(len(row_counter))
            plt.bar(x, row_counter.values())
            plt.xticks(x, row_counter.keys())
        elif column_name in numeric_histogram_columns_with_bins:
            plt.hist(df[column_name].values,
                     numeric_histogram_columns_with_bins[column_name])
        elif column_name in violin_plots:
            sns.violinplot(data=df[column_name].values)
        else:
            plt.hist(df[column_name].values, bins=50)

        if column_name == PRECIPITATION:
            # plt.title("{} LINEAR SCALE".format(column_name))

            plt.ylabel("Number of flights (linear scale)")
            plt.xlabel(lables[column_name])

            index_in_suplot += 1
            plt.subplot(plt_rows, plt_columns, index_in_suplot)
            plt.hist(df[column_name].values, bins=50)
            # plt.title("PRECIPITATION LOGARITHMIC SCALE")
            plt.yscale("log")

            plt.ylabel("Number of flights (logaritmic scale)")
            plt.xlabel(lables[column_name])
        else:
            # plt.title(column_name)
            plt.ylabel("Number of flights")
            plt.xlabel(lables[column_name])

    for column_name in non_numeric_columns:
        index_in_suplot += 1
        plt.subplot(plt_rows, plt_columns, index_in_suplot)
        row_counter = Counter(df[column_name].values)
        x = np.arange(len(row_counter))
        plt.bar(x, row_counter.values())
        plt.xticks(x, row_counter.keys())
        # plt.title(column_name)

        plt.ylabel("Number of flights")
        plt.xlabel(lables[column_name])

    plt.tight_layout()
    save_plt_figure(fig, location, "plot_per_column")
    # plt.show()
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
    # plt.show()
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
    # plt.show()
    plt.close()


def target_distribution(df, location):
    fig = plt.figure(figsize=(8, 6))
    distribution = []
    for i in range(len(TARGET_CLASSES)):
        subset_df = df[df[TARGET] == i]
        subset_df = subset_df[TARGET]
        distribution.append(subset_df.count())

    plt.bar(TARGET_CLASSES, distribution)
    plt.title(location)
    plt.xlabel("Target")
    plt.ylabel("Number of flights")
    save_plt_figure(fig, location, "target_distribution")
    # plt.show()
    plt.close()


# ! THESE ARE BROKEN. Only 2 targets are available.
def target_per_column(df, column_name, filename_prefix, location):
    fig = plt.figure(figsize=(8, 8))
    for i in range(len(TARGET_CLASSES_THREE)):
        subset_df = df[df[TARGET] == i]
        subset_df = subset_df[[column_name, TARGET]]
        values = subset_df.groupby(subset_df[column_name]).size()
        plt.plot(values)

    plt.title(location)
    plt.legend(TARGET_CLASSES_THREE)
    plt.xlabel(column_name)
    plt.ylabel("Count")
    save_plt_figure(fig, location, "{}_per_target".format(filename_prefix))
    # plt.show()
    plt.close()


def target_per_hour(df, location):
    target_per_column(df, HOUR, "hourly", location)


def target_per_month(df, location):
    target_per_column(df, MONTH, "monthly", location)


if __name__ == "__main__":
    main()
