import argparse
import pandas as pd
import requests
import re
from io import StringIO
import os

# API credentials and config
CLIENT_ID = "331fd22f-e045-4e6f-bfcd-c5562770e815"
CLIENT_SECRET = "6a8bf10a-0f28-44e1-9bc0-babac1c975be"
API_URL = "https://frost.met.no/observations/v0.csv"

# Frost location IDs
GARDERMOEN_SOURCE_ID = "SN4780"
VAERNES_SOURCE_ID = "SN69100"

# File names
RAW_PATH = "../../../data/raw/weather"
RAW_FILE_NAME_AIR_TEMPERATURE = "air_temperature.csv"
RAW_FILE_NAME_PRECIPITATION = "precipitation.csv"
RAW_FILE_NAME_VISIBILITY = "visibility.csv"
RAW_FILE_NAME_WIND_DIRECTION = "wind_direction.csv"
RAW_FILE_NAME_WIND_SPEED = "wind_speed.csv"


def main():
    args = parse_args()

    time_interval = args.date_interval
    time_interval = time_interval[:10] + \
        "T00:00:00.000Z" + time_interval[10:] + "T23:59:59Z"

    request_builder = newRequestBuilder(args.source_id, time_interval)
    saveFn = newFileSaver(args.source_id)

    getWindSpeed(request_builder, saveFn)
    getWindDirection(request_builder, saveFn)
    getAirTemp(request_builder, saveFn)
    getRainfall(request_builder, saveFn)
    getVisibility(request_builder, saveFn, args.source_id)


def parse_args():
    """Returns the command line arguments."""

    parser = argparse.ArgumentParser(
        description="Fetch some observations."
    )
    parser.add_argument("date_interval",
                        help="e.g. 2020-01-01/2020-05-05"
                        )
    parser.add_argument("-s", "--source_id",
                        default="SN4780",
                        help="Default Gardermoen. Source ID of the desired location from api.met.no e.g. SN4780"
                        )

    return parser.parse_args()


def getWindSpeed(request_builder, save):
    """Fetches, parses, and saves wind speed data.

    Args:
        request_builder (function): The request builder.
        save (function): The save function to run to save the data.

    """

    req = request_builder("wind_speed", "PT20M", "PT30M")
    df = runRequest(req)

    save(df, RAW_FILE_NAME_WIND_SPEED)


def getWindDirection(request_builder, save):
    """Fetches, parses, and saves wind direction data.

    Args:
        request_builder (function): The request builder.
        save (function): The save function to run to save the data.

    """

    req = request_builder("wind_from_direction", "PT20M", "PT30M")
    df = runRequest(req)

    save(df, RAW_FILE_NAME_WIND_DIRECTION)


def getAirTemp(request_builder, save):
    """Fetches, parses, and saves air temperature data.

    Args:
        request_builder (function): The request builder.
        save (function): The save function to run to save the data.

    """

    req = request_builder("air_temperature", "PT20M", "PT30M")
    df = runRequest(req)

    save(df, RAW_FILE_NAME_AIR_TEMPERATURE)


def getRainfall(request_builder, save):
    """Fetches, parses, and saves rainfall data.

    Args:
        request_builder (function): The request builder.
        save (function): The save function to run to save the data.

    """

    req = request_builder("accumulated(precipitation_amount)", "PT0H", "PT1H")
    req["referencetime"] = re.sub('T.+?Z', '', req["referencetime"])
    df = runRequest(req)

    save(df, RAW_FILE_NAME_PRECIPITATION)


def getVisibility(request_builder, save, source_id):
    """Fetches, parses, and saves visibility data.

    Args:
        request_builder (function): The request builder.
        save (function): The save function to run to save the data.
        source_id (str): The source id for the location.

    """

    req = None

    if source_id == GARDERMOEN_SOURCE_ID:
        req = request_builder(
            "visibility_in_air_poorest_direction", "PT0H", "PT6H")
    elif source_id == VAERNES_SOURCE_ID:
        req = request_builder("visibility_in_air_mor", "PT0H", "PT1H")
    else:
        return

    df = runRequest(req)
    print(df)

    save(df, RAW_FILE_NAME_VISIBILITY)


def runRequest(request):
    """Fetches a request and returns the data

    Args:
        request (dict): A dict containing request parameters.

    Returns:
        pandas DataFrame: The fetched data as a dataframe.

    """

    url = "https://frost.met.no/observations/v0.csv"
    print("Downloading %s..." % request['elements'])
    response = requests.get(url, request, auth=(CLIENT_ID, ''))

    if(response.status_code != 200):
        print("Error! Returned status code {} for the parameter: {}".format(
            response.status_code, request['elements']))
        print("Message: {}".format(response.json()['error']['message']))
        print("Reason: {}".format(response.json()['error']['reason']))
        return None

    df = pd.read_csv(StringIO(response.text))

    if df is None:
        print("There was an error")
        os.exit(1)

    return df


def newRequestBuilder(sources, time_interval):
    """Creates a new request builder

    Args:
        sources (str): The sources parameter for Frost.
        time_interval (str): The time interval parameter for Frost.

    Returns:
        function: A request builder.

    """

    def builder(element, offset, interval):
        """Builds a Frost request in the current source and time interval scope.

        Args:
            element (str): The element parameter for Frost.
            offset (str): The offset parameter for Frost.
            interval (str): The interval parameter for Frost.

        Returns:
            dict: The Frost request.

        """

        return {
            'sources': sources,
            'elements': element,
            'referencetime': time_interval,  # note that this != column_names.reference_time
            'timeoffsets': offset,
            'timeresolutions': interval,
        }

    return builder


def newFileSaver(source_id):
    """Creates a new file saver

    Args:
        source_id (str): The directory name for the files.

    Returns:
        function: A save function to save a dataframe to the directory.

    """

    def save(df, file_name):
        location = source_id
        if(location == GARDERMOEN_SOURCE_ID):
            location = "OSL"
        elif(location == VAERNES_SOURCE_ID):
            location = "TRD"

        file_dir = '%s/%s' % (RAW_PATH, location)

        if not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        df.to_csv('%s/%s' % (file_dir, file_name), index=False)

    return save


if __name__ == "__main__":
    main()
