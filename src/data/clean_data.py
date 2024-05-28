import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helper import *


# rider = 'tadej-pogacar'
def race(rider):
    race_df = pd.read_csv(f'data/raw/riders/{rider}.csv')
    print(race_df.head())


# Basic data cleaning
def data_cleaning(verbose=False):
    riders_info = pd.read_csv('data/raw/riders_info.csv')

    # 1e6 = missing value for the rankings
    riders_info['PCS rank'] = riders_info['PCS rank'].fillna(1e6).astype(int)
    riders_info['UCI rank'] = riders_info['UCI rank'].fillna(1e6).astype(int)
    riders_info['All time rank'] = riders_info['All time rank'].fillna(1e6).astype(int)

    # convert DOB to datetime
    riders_info['DOB'] = pd.to_datetime(riders_info['DOB'])

    riders_info = riders_info.rename(columns={'weight [kg]': 'weight', 'height [m]': 'height'})

    # filter outliers
    riders_info = riders_info[riders_info['weight'] < 100]
    riders_info = riders_info[riders_info['height'] > 1]

    if verbose:
        print(riders_info.info())
        print(riders_info.head())

    riders_info.to_csv('data/processed/riders_info_cleaned.csv', index=False)


# Find long and lat for birth places
def find_birth_places():
    riders_info = pd.read_csv('data/processed/riders_info_cleaned.csv')

    place_map = [] # list of tuples (address, lon, lat)
    place_map_fail = []
    for bp in riders_info['birth place'].dropna().unique():
        place = bp.strip()
        try:
            x,y = get_lon_lat(place)
            print(f'Found lon {x} and lat {y} for {place}\n')
            place_map.append((place, x, y))
        except:
            print(f'cannot find lon and lat for {place}\n')
            place_map_fail.append(place)
        time.sleep(1.5) # to avoid getting blocked by the server

    print('\n\nDone searching....')
    place_map_df = pd.DataFrame(place_map, columns=['address', 'lon', 'lat'])
    place_map_df.to_csv('data/processed/place_map.csv', index=False)

    print('\nThese are the failed places:')
    print(place_map_fail)
    with open('data/processed/place_map_fail.txt', 'w') as f:
        for place in place_map_fail:
            f.write(f'{place}\n')


# Create new columns (lon, lat) of birth place
def attach_birth_place_long_lat():
    riders_info = pd.read_csv('data/processed/riders_info_cleaned.csv')


    places = pd.read_csv('data/processed/place_map.csv')
    def finding_lon_lat(place):
        for i in range(len(places)):
            if place == places.iloc[i]['address']:
                return (places.iloc[i]['lon'], places.iloc[i]['lat'])
        return None, None

    riders_info['lonlat'] = riders_info['birth place'].apply(lambda x: finding_lon_lat(x))
    riders_info.to_csv('data/processed/riders_info_cleaned.csv', index=False)


# date='00.04.1996', '26.08.2020', '00.06.2010'
def fix_zero_date(row):
    if len(row) == 0:
        return
    distance = row['distance']

    date = row['date']
    dd = int(date.split('.')[0])
    mm = int(date.split('.')[1])
    yy = int(date.split('.')[2])

    if mm == 6 and yy == 2010 and distance == 226.4:
        # https://www.procyclingstats.com/race/nc-spain/2010/result
        return '27.06.2010'

    if dd == 0 and mm == 0:
        if yy in [1986, 1987, 1989, 1993, 1995]:
            return
        return date # manually handle this

    if dd == 0:
        return f'01.{mm}.{yy}'

    return date


def data_cleaning_single_rider(name):
    rider_race = pd.read_csv(f'data/raw/riders/{name}.csv')
    rider_race = rider_race[['date', 'result ranking', 'race name', 'distance', 'year']]

    # drop the date range, like "16.06 » 19.06"
    rider_race = rider_race[rider_race['date'].astype(str).map(len) <= 5]
    rider_race = rider_race.dropna(subset=['date'])

    # append the year
    rider_race['date'] = rider_race['date'].astype(str) + '.' + rider_race['year'].astype(str)
    rider_race = rider_race.drop(columns=['year'])

    # fix zero date
    rider_race['date'] = rider_race.apply(fix_zero_date, axis=1)

    # convert to datetime
    rider_race['date'] = pd.to_datetime(rider_race['date'], format='%d.%m.%Y')

    rider_race.to_csv(f'data/processed/riders/{name}.csv', index=False)


def data_clean_all_riders():
    riders = os.listdir('data/raw/riders') # ['primoz-roglic.csv', 'tadej-pogacar.csv', 'alejandro-valverde.csv' ...]

    for rider in riders:
        rider_name = rider[:-4] # remove .csv
        data_cleaning_single_rider(rider_name)


def riders_processed_data_parquet():
    """
    Convert data/processed/riders/{name}.csv to pyspark parquet format
    takes ~6min
    """
    from pyspark.sql import SparkSession
    from pyspark.testing import assertDataFrameEqual

    spark = SparkSession.builder.master("local[*]").config("spark.driver.memory", "4g").getOrCreate()
    riders = os.listdir('data/processed/riders')
    for idx, rider in enumerate(riders):
        rider_name = rider[:-4] # remove .csv
        rider_df = pd.read_csv(f'data/processed/riders/{rider_name}.csv')
        rider_df['rider'] = rider_name
        all_riders = pd.concat([all_riders, rider_df]) if idx > 0 else rider_df # concat all riders

    riders_df = spark.createDataFrame(all_riders)
    riders_df.write.mode("overwrite").parquet("data/processed/riders.parquet")

    # check if saving and reading is correct
    riders_df_read = spark.read.load("data/processed/riders.parquet")
    assertDataFrameEqual(riders_df, riders_df_read)
    print('Done')


if __name__ == "__main__":
    # data_cleaning()
    # find_birth_places()
    # attach_birth_place_long_lat()

    # data_clean_all_riders()
    riders_processed_data_parquet()
