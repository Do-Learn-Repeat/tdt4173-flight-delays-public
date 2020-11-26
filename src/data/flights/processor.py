import math
import pytz
from datetime import datetime, date, time, timezone
import numpy as np
import pandas as pd
import os

INPUT_PATH = '../../../data/raw/flights'
OUTPUT_PATH = '../../../data/processed/flights'


def main():
    df = pd.DataFrame()
    # Commented out due to only providing sample of 2015 data
    # for year in [2015, 2016, 2017, 2018, 2019, 2020]:
    for year in [2016]:
        print("Loading flights for year %s" % year)
        data = pd.read_excel('%s/20201020_NTNU__Rute_%s.xlsx' %
                             (INPUT_PATH, year))
        df = pd.concat([df, data], ignore_index=True)

    # remove flights that have no scheduled time
    df = df[df.SOBT_SIBT_UTC.notnull()]
    # remove flights which are not cancelled AND have no actual departure/arrival time at gate
    df = df[(df.AOBT_AIBT_UTC.notnull()) | (
        df.AOBT_AIBT_UTC.isnull() & df.CANCELLED.notnull())]

    print('Total shape:', df.shape)

    df['SOBT_SIBT'] = df['SOBT_SIBT_UTC'].apply(datetime_parser)
    df['ATOT_ALDT'] = df['ATOT_ALDT_UTC'].apply(datetime_parser)
    df['AOBT_AIBT'] = df['AOBT_AIBT_UTC'].apply(datetime_parser)
    df.drop(columns=['SOBT_SIBT_UTC', 'ATOT_ALDT_UTC',
                     'AOBT_AIBT_UTC'], inplace=True)

    df['WEEKDAY'] = df['SOBT_SIBT'].apply(get_weekdaynumber)

    # Send each row into the apply function
    df['DELAY'] = df.apply(get_delay, axis=1)

    # Export csv
    dfs = dict(tuple(df.groupby('AIRPORT')))

    for (key, frame) in dfs.items():
        frame.drop(columns=['AIRPORT'], inplace=True)

        file_dir = '%s/%s' % (OUTPUT_PATH, key)

        os.makedirs(file_dir, exist_ok=True)

        # Save small sample
        frame[0:100].to_csv(file_dir+'/processed_sample.csv', index=False)

        # Save complete dataset in chunks
        chunk, chunk_size = 0, 500000
        while chunk < len(frame):
            frame[chunk:chunk+chunk_size].to_csv('%s/processed_part%s.csv' % (
                file_dir, int(chunk/chunk_size)), index=False)
            chunk += chunk_size


def get_weekdaynumber(datetime_iso):
    '''
    Extract weekday<number> 
    '''
    return datetime.fromisoformat(datetime_iso).weekday()


def get_delay(row):
    '''
    Extract delaytime
    '''
    # return inf if flight is cancelled
    if isinstance(row['CANCELLED'], str) and row['CANCELLED'] == 'C':
        return math.inf

    scheduled = datetime.fromisoformat(row['SOBT_SIBT'])
    actual = datetime.fromisoformat(row['AOBT_AIBT'])
    time_delta = actual - scheduled
    minutes = time_delta.total_seconds()/60
    return minutes


def datetime_parser(raw_time_object):
    '''
    Extract date+time in SOBT_SIBT_UTC, ATOT_ALDT_UTC, AOBT_AIBT_UTC columns and convert to correct timezone
    '''
    if isinstance(raw_time_object, str):
        # raw_time_object.split() => [date, time]
        raw_timedata = raw_time_object.split()

        year = int(raw_timedata[0][6:10])
        month = int(raw_timedata[0][3:5])
        day = int(raw_timedata[0][0:2])
        hour = int(raw_timedata[1][0:2])
        minute = int(raw_timedata[1][3:5])
        utc = pytz.UTC
        date_time = datetime(year, month, day, hour, minute, tzinfo=utc)

        # convert time to Norway time, and remove timezone information
        norway_timezone = pytz.timezone('Europe/Oslo')
        date_time = date_time.astimezone(norway_timezone)
        date_time = date_time.replace(tzinfo=None)
        date_time = date_time.isoformat()
        return date_time
    return math.nan


if __name__ == "__main__":
    main()
