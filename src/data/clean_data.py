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


if __name__ == "__main__":
    data_cleaning()
    find_birth_places()
    attach_birth_place_long_lat()