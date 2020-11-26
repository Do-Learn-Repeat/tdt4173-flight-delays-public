import pandas as pd
import os
import numpy as np

WEATHER_PROCESSED_PATH = "../../../data/processed/weather"
FLIGHTS_PROCESSED_PATH = "../../../data/processed/flights"

# Paths
CLEAN_PATH = "../../../data/clean"


def main():
    for location in os.listdir(FLIGHTS_PROCESSED_PATH):
        if os.path.isdir("%s/%s" % (FLIGHTS_PROCESSED_PATH, location)):
            # ignore files to skip READMEs etc
            process(location)


def process(location):
    """Processes and saves a dataset for a location.

    Args:
        location (Location): The location to process.

    """

    print("\n\n> Processing %s\n" % location)

    # Load data
    flights = loadFlightData(location)
    weather = loadWeatherData(location)

    # Remove bad data
    flights = removeCancelledFlights(flights)
    flights = removeOutliersOutOfRange(flights, 'DELAY', 500)

    # Remove flights outside of weather coverage
    flights = flights[flights['SOBT_SIBT'] > weather.iloc[0]['REFERENCETIME']]
    flights = flights[flights['SOBT_SIBT'] < weather.iloc[-1]['REFERENCETIME']]

    # Merge and ensure data
    df = mergeData(flights, weather)
    if df.empty:
        return

    # Process dataframe (add targets, time-features and clean the DF)
    df['TARGET'] = df['DELAY'].apply(calculateTarget)

    df['HOUR'], df['DAY_OF_MONTH'], df['MONTH'], df['YEAR'] = getTimeFeatures(
        df['SOBT_SIBT'])

    # Clean final data
    df = cleanDF(df)

    print('Saving dataset...')
    df.to_csv("%s/%s.csv" % (CLEAN_PATH, location), index=False)


# Synthetic sugar for readability
def removeCancelledFlights(data):
    """Removes cancelled flights from a dataframe.

    Args:
        data (pandas DataFrame): The dataframe to remove from.

    """

    return data[np.isfinite(data['DELAY'])]


def removeOutliersOutOfRange(data, key, max_range):
    return data[(data[key] < max_range) & (data[key] > -max_range)]


def mergeData(flights, weather):
    """Joins 'close enough' weather data on flights by scheduled time.

    Args:
        flights (pandas DataFrame): The dataframe containing flight records.
        weather (pandas DataFrame): The dataframe containing weather records.

    Returns:
        pandas DataFrame: The merged/joined data.

    """

    i = 0       # variable for printing progress
    data = []   # Merged data for this location
    nextWeather = weatherIterator(weather)

    for _, flight_record in flights.iterrows():
        weather_record = nextWeather(flight_record)

        # Merge weather and flight
        row = flight_record.to_dict()
        row.update(weather_record.to_dict())

        # Append merged data
        data.append(row)

        # Print is an expensive operation.
        # So we'll only print every 1000th iteration.
        if i % 1000 == 0:
            print('Merging data (%s/%s)\r' % (i, len(flights)), end=""),
        i += 1

    print('Data is merged!                             \n')

    return pd.DataFrame(data)


def weatherIterator(weather):
    """
    Returns an iterative function that returns the weather closest
    to a given flight record.

    NOTE: This is an iterator and assumes that the flight and weather 
    records are in chronological order!

    Args:
        flights (pandas DataFrame): The dataframe containing flight records.
        weather (pandas DataFrame): The dataframe containing weather records.

    Returns:
        function: Iterative function retrieving correct weather for the next flight.

    """

    wi = 0
    num_weather_entries = len(weather)

    def iterator(flight_record):
        """Returns the closest weather data for the provided flight record."""

        nonlocal wi
        flight_time = flight_record['SOBT_SIBT']

        # Loop until wi is the data point closest to the flight record
        # (while not out-of-bounds and the next datapoint is closer than the current)
        while wi+1 < num_weather_entries and abs(flight_time - weather.iloc[wi+1]['REFERENCETIME']) < abs(flight_time - weather.iloc[wi]['REFERENCETIME']):
            wi += 1

        return weather.iloc[wi]

    return iterator


def loadFlightData(location):
    """Loads all flight data for a specific location and returns it as a dataframe.

    Args:
        location (Location): The location to load data from.

    Returns:
        pandas DataFrame: A dataframe containing the flight records

    """

    print("Loading flight data")

    path = "%s/%s" % (FLIGHTS_PROCESSED_PATH, location)
    df = pd.DataFrame()
    for part in os.listdir(path):
        if "sample" in part:
            continue  # skip samples

        data = pd.read_csv('%s/%s' % (path, part))
        df = pd.concat([df, data], ignore_index=True)

    df['SOBT_SIBT'] = pd.to_datetime(df['SOBT_SIBT'])
    df['ATOT_ALDT'] = pd.to_datetime(df['ATOT_ALDT'])
    df['AOBT_AIBT'] = pd.to_datetime(df['AOBT_AIBT'])

    df = df.sort_values(by=['SOBT_SIBT'])
    df.reset_index(inplace=True)

    print("Flight data loaded!\n")
    return df


def loadWeatherData(location):
    """Loads all weather data for a specific location and returns it as a dataframe.

    Args:
        location (Location): The location to load data from.

    Returns:
        pandas DataFrame: A dataframe containing the weather records

    """

    print("Loading weather data")
    df = pd.read_csv("%s/%s.csv" % (WEATHER_PROCESSED_PATH, location))

    df['REFERENCETIME'] = pd.to_datetime(df['REFERENCETIME'])

    df = df.sort_values(by=['REFERENCETIME'])
    df.reset_index(inplace=True)

    print("Weather data loaded!\n")
    return df


def calculateTarget(delay):
    """Classifies a delay time as either on-time (0) or delayed (1).

    Args:
        delay (int): How much a flight was delayed by. Negative numbers is before scheduled.

    Returns:
        int: The numeric representation of the classification (0 = on time, 1 = delayed).

    """

    # The official standard for when a flight is 'delayed'
    # is 15 minutes, as proposed by the FFA
    # https://en.wikipedia.org/wiki/Flight_cancellation_and_delay
    delay_margin = 15

    if delay < delay_margin:
        # the flight departured before 15 minutes after the
        # scheduled time, and is, therefore, on time.
        return 0

    # The flight is over 15 minutes late and is, therefore, delayed.
    return 1


def getTimeFeatures(list_of_dt):
    """Extracts time elements from a list of datetimes.

    Args:
        list_of_dt ([datatime]): A list of datetime.

    Returns:
        list: A list of pandas Series [hours, days, months, years]

    """

    hours = []
    days = []
    months = []
    years = []

    for dt in list_of_dt:
        hours.append(dt.hour)
        days.append(dt.day)
        months.append(dt.month)
        years.append(dt.year)

    return [pd.Series(hours), pd.Series(days), pd.Series(months), pd.Series(years)]


def cleanDF(df):
    """
    Deletes common unused fields in the dataset
    (REFERENCETIME, CANCELLED, SOBT_SIBT, ATOT_ALDT, AOBT_AIBT, DELAY, index)
    and reorders columns.

    Args:
        df (pandas DataFrame): The dataframe to clean.

    Returns:
        pandas DataFrame: The cleaned dataframe.

    """

    del df['REFERENCETIME']
    del df['CANCELLED']
    del df['SOBT_SIBT']
    del df['ATOT_ALDT']
    del df['AOBT_AIBT']
    del df['DELAY']
    del df['index']

    # Reorder the columns in df, NB: if some of the above is decided to not be deleted, then it must be added in this list
    df = df[['YEAR', 'MONTH', 'DAY_OF_MONTH', 'WEEKDAY', 'HOUR', 'DEP_ARR', 'AIRLINE_IATA', 'FLIGHT_ID', 'TO_FROM',
             'INT_DOM_SCHENGEN', 'GATE_STAND', 'PRECIPITATION', 'WIND_DIRECTION', 'VISIBILITY', 'AIR_TEMPERATURE', 'WIND_SPEED', 'TARGET']]

    return df


if __name__ == "__main__":
    main()
