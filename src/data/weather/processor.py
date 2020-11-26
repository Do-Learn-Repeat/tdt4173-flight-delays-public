import os
import pandas as pd
import dateutil.parser
from datetime import datetime, timedelta
import copy

from scraper import (
    RAW_FILE_NAME_AIR_TEMPERATURE,
    RAW_FILE_NAME_PRECIPITATION,
    RAW_FILE_NAME_VISIBILITY,
    RAW_FILE_NAME_WIND_DIRECTION,
    RAW_FILE_NAME_WIND_SPEED,

    RAW_PATH,

    GARDERMOEN_SOURCE_ID,
    VAERNES_SOURCE_ID
)

# Keys
REFERENCE_TIME = "referenceTime"
WIND_SPEED = "wind_speed"
WIND_DIRECTION = "wind_direction"
AIR_TEMPERATURE = "air_temperature"
VISIBILITY = "visibility"
PRECIPITATION = "precipitation"

# Paths
PROCESSED_PATH = "../../../data/processed/weather"


def main():
    # Create the output dir
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    # parse_map contains per-file parsers. For example, visibility is parsed differently
    # from air temperature. The key of the map are the file names which are attached
    # to their respective parsing functions.
    parse_map = {
        RAW_FILE_NAME_AIR_TEMPERATURE: parseAirTemperature,
        RAW_FILE_NAME_PRECIPITATION: parsePrecipitation,
        RAW_FILE_NAME_VISIBILITY: parseVisibility,
        RAW_FILE_NAME_WIND_DIRECTION: parseWindDirection,
        RAW_FILE_NAME_WIND_SPEED: parseWindSpeed,
    }

    # RAW_PATH is a dir containing folders of location (e.g. OSL, TRD).
    # These locations are dirs containing weather info (e.g. air_temperature.csv etc)
    for location in os.listdir(RAW_PATH):
        print("\nProcessing weather data for %s" % location)

        # Data is a dict containing parsed dataframes for each weather category
        # data['air_temperature'] = pd.Dateframe
        data = {}
        for file_name in os.listdir('%s/%s' % (RAW_PATH, location)):
            df = pd.read_csv('%s/%s/%s' % (RAW_PATH, location, file_name))

            print('Parsing %s' % file_name)
            parser = parse_map[file_name]   # Get appropriate parser for file

            # Parse data (key can for example be AIR_TEMPERATURE)
            (df, key) = parser(df)

            # Do common parsing for all frames
            dropExcessColumns(df)
            df[REFERENCE_TIME] = df.apply(removeTimeZone, axis=1)

            # Attach dataframe to the data object
            data[key] = df

        if data:
            # Data should now be a dict containing data about
            # all weather categories for this location. Let's merge them
            # and save the output as <location>.csv
            print("Merging data...")
            merged = mergeDataFrames(data)
            merged.to_csv('%s/%s.csv' %
                          (PROCESSED_PATH, location), index=False)


def parseAirTemperature(df):
    """Parses air temperature from a dataframe.

    Args:
        df (pandas DataFrame): The dataframe to parse.

    Returns:
        (pandas DataFrame, str): [0] is the parsed dataframe and [1] is the key to what was parsed.

    """

    df.drop('height_above_ground(m)', axis=1, inplace=True)
    df = shiftTime(df, 10)
    renameColumn(df, "air_temperature(degC)", AIR_TEMPERATURE)

    return (df, AIR_TEMPERATURE)


def parsePrecipitation(df):
    """Parses precipitation from a dataframe.

    Args:
        df (pandas DataFrame): The dataframe to parse.

    Returns:
        (pandas DataFrame, str): [0] is the parsed dataframe and [1] is the key to what was parsed.

    """

    renameColumn(df, "accumulated(precipitation_amount)(mm)", PRECIPITATION)
    df = fixPrecipitation(df)
    df = ensure30MinuteIntervals(df)

    return (df, PRECIPITATION)


def parseVisibility(df):
    """Parses visibility from a dataframe.

    Args:
        df (pandas DataFrame): The dataframe to parse.

    Returns:
        (pandas DataFrame, str): [0] is the parsed dataframe and [1] is the key to what was parsed.

    """

    # OSL and TRD are treated differently as their visibility data is different.
    # This function will panic for other locations than OSL and TRD.
    # Returns the parsed dataframe and VISIBILITY key.

    source_id = df['sourceId'].values[0]

    if source_id.upper().startswith(GARDERMOEN_SOURCE_ID):
        renameColumn(
            df, "visibility_in_air_poorest_direction(meters)", VISIBILITY)
    elif source_id.upper().startswith(VAERNES_SOURCE_ID):
        renameColumn(df, "visibility_in_air_mor(meters)", VISIBILITY)
    else:
        print(
            "No parsing-implementation for visibility on other locations than OSL and TRD")
        os.exit(1)

    df = ensure30MinuteIntervals(df)
    return (df, VISIBILITY)


def parseWindDirection(df):
    """Parses wind direction from a dataframe.

    Args:
        df (pandas DataFrame): The dataframe to parse.

    Returns:
        (pandas DataFrame, str): [0] is the parsed dataframe and [1] is the key to what was parsed.

    """

    df.drop('height_above_ground(m)', axis=1, inplace=True)
    df = shiftTime(df, 10)
    renameColumn(df, "wind_from_direction(degrees)", WIND_DIRECTION)

    return (df, WIND_DIRECTION)


def parseWindSpeed(df):
    """Parses wind speed from a dataframe.

    Args:
        df (pandas DataFrame): The dataframe to parse.

    Returns:
        (pandas DataFrame, str): [0] is the parsed dataframe and [1] is the key to what was parsed.

    """

    df.drop('height_above_ground(m)', axis=1, inplace=True)
    df = shiftTime(df, 10)
    renameColumn(df, "wind_speed(m/s)", WIND_SPEED)

    return (df, WIND_SPEED)


def mergeDataFrames(dfs):
    """Merges all dataframes inside of the provided dict.

    Args:
        dfs (dict): A dict where {[data_key]: dataframe}.

    Returns:
        pandas DataFrame: The merged dataframe.

    """

    prev_df = None
    for df in dfs.values():
        if prev_df is None:
            prev_df = df
        else:
            prev_df = prev_df.merge(
                df, on=REFERENCE_TIME, how='inner')

    prev_df.columns = prev_df.columns.str.upper()

    return prev_df


def dropExcessColumns(df):
    """
    Removes excess/unused columns
    (sourceId, timeOffset, timeResolution, performanceCategory,
    exposureCategory, qualityCode, and timeSeriesId from a dataframe).

    Args:
        df (pandas DataFrame): The dataframe to remove columns in.

    """

    df.drop('sourceId', axis=1, inplace=True, errors='ignore')
    df.drop('timeOffset', axis=1, inplace=True, errors='ignore')
    df.drop('timeResolution', axis=1, inplace=True, errors='ignore')
    df.drop('performanceCategory', axis=1, inplace=True, errors='ignore')
    df.drop('exposureCategory', axis=1, inplace=True, errors='ignore')
    df.drop('qualityCode', axis=1, inplace=True, errors='ignore')
    df.drop('timeSeriesId', axis=1, inplace=True, errors='ignore')


def removeTimeZone(row):
    """
    Removes the timezone attribute from the REFERENCE_TIME column in
    the provided dataframe.

    Args:
        row (pandas DataFrame row): A row of a dataframe to remove time zone in.

    Returns:
        str: The date without timezone.

    """
    # df.iterrows() was initially tried here, but a hefty bug occured. Keep this.
    date = dateutil.parser.isoparse(row[REFERENCE_TIME])
    date = date.replace(tzinfo=None)
    return datetime.isoformat(date)


def shiftTime(df, minutes_to_shift):
    """Maps referenceTime :20 to :30 and :50 to :00, so that it can be more easily merged.

    Args:
        df (pandas DataFrame): The dataframe to shift.
        minutes_to_shift (int): How much the df should be shifted by in minutes.

    Returns:
        pandas DataFrame: The shifted dataframe.

    """

    # TODO Maybe replace iterrows with lambda or numpy for speed improvements
    for index, row in df.iterrows():
        date = dateutil.parser.isoparse(row[REFERENCE_TIME])
        date += timedelta(minutes=minutes_to_shift)
        df.at[index, REFERENCE_TIME] = datetime.isoformat(date)
    return df


def renameColumn(df, old_name, new_name):
    """Renames a column in a provided dataframe.

    Args:
        df (pandas DataFrame): The dataframe containing the column to rename.
        old_name (str): The name of the column to rename.
        new_name (str): The new name of the column to rename.

    """

    df.rename(columns={old_name: new_name}, inplace=True)


# This function is commented out as it lead to
# many unwanted duplicates. We might want to
# look into it later to segregate sections by
# each side of an index instead of on the
# right of an index.

def ensure30MinuteIntervals(df):
    """
    Extends the rows where they are missing. For example, visibility every 3rd hour is now every 30 min.
    The rows are extended both upwards and downwards with 30. min intervals

    Args:
        df (pandas DataFrame): The dataframe to ensure correct intervals in.

    Returns:
        pandas DataFrame: The parsed dataframe.

    """

    firstTime = dateutil.parser.isoparse(df.at[0, REFERENCE_TIME])
    secondTime = dateutil.parser.isoparse(df.at[1, REFERENCE_TIME])
    timeInterval = secondTime - firstTime
    minuteInterval = int(timeInterval.seconds / 60)

    extendRange = int(minuteInterval / 30)

    if(extendRange < 2):
        return df

    lower = (extendRange // 2)
    upper = extendRange - lower
    data = []

    for _, row in df.iterrows():
        row_time = dateutil.parser.isoparse(row[REFERENCE_TIME])
        for i in range(-lower + 1, upper + 1):
            if i == 0:  # Avoids duplicating the original row
                continue
            newRow = copy.deepcopy(row)
            new_time = row_time + timedelta(minutes=(30 * i))
            new_time = new_time.replace(tzinfo=None)
            new_time = datetime.isoformat(new_time)
            newRow[REFERENCE_TIME] = new_time
            data.append(newRow)

    return df.append(data)


def fixPrecipitation(df):
    """
    Fixes precipitation in a provided dataframe.
    Precipitation is presented in the API as accumulated values
    as new_value = prev_accumulated + new_addition_precipitation.
    For example, instead of (1, 1, 1, 1 - 1mm every hour) the data is
    (1, 2, 3, 4).

    Args:
        df (pandas DataFrame): The dataframe to fix precipitation in.

    Returns:
        pandas DataFrame: The dataframe with fixed precipitation.

    """

    prev_value = df.at[0, PRECIPITATION]
    for index, row in df.iterrows():
        initial_value = row[PRECIPITATION]
        if(initial_value < prev_value):
            df.at[index, PRECIPITATION] = 0.0
        else:
            value = round(df.at[index, PRECIPITATION] - prev_value, 3)
            df.at[index, PRECIPITATION] = value
        prev_value = initial_value

    return df


if __name__ == "__main__":
    main()
