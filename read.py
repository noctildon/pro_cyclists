import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from helper import *
from ast import literal_eval


# rider = 'tadej-pogacar'
def race(rider):
    race_df = pd.read_csv(f'riders/{rider}.csv')
    print(race_df.head())


def data_cleaning(verbose=False):
    riders_info = pd.read_csv('riders_info.csv')

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

    return riders_info


# weight/height plot
def hw_plot():
    riders_info = data_cleaning()

    # histogram of weight
    riders_info.hist(bins=50, column=['weight', 'height'])
    plt.show()

    # scatter plot
    plt.title('Riders weight and height')
    sns.scatterplot(x='weight', y='height', data=riders_info)
    plt.xlabel('Weight [kg]')
    plt.ylabel('Height [m]')
    plt.show()


# Find long and lat for birth places
def find_birth_places():
    riders_info = data_cleaning()

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
    place_map_df.to_csv('place_map.csv', index=False)

    print('\nThese are the failed places:')
    print(place_map_fail)
    with open('place_map_fail.txt', 'w') as f:
        for place in place_map_fail:
            f.write(f'{place}\n')


# Create new columns (lon, lat) for birth place
def attach_birth_place_long_lat():
    riders_info = data_cleaning()

    places = pd.read_csv('place_map.csv')
    def finding_lon_lat(place):
        for i in range(len(places)):
            if place == places.iloc[i]['address']:
                return (places.iloc[i]['lon'], places.iloc[i]['lat'])
        return None, None

    riders_info['lonlat'] = riders_info['birth place'].apply(lambda x: finding_lon_lat(x))
    riders_info.to_csv('riders_info_cleaned.csv', index=False)


# plot geographical birth place
# sortby = 'UCI rank', 'weight', 'dob', etc
# numbers = how many riders to plot
def birth_place_plot(sortby='UCI rank', numbers=100, ascending=True):
    riders_info = pd.read_csv('riders_info_cleaned.csv')

    plt.figure(figsize=(8, 8))
    m = Basemap(projection='cyl', resolution=None, llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180,lat_0=0, lon_0=0)
    # m = Basemap(projection='moll', resolution=None, lat_0=0, lon_0=0) # for mollweide projection
    m.etopo(scale=0.5, alpha=0.5)

    # sort the riders
    riders_info = riders_info.sort_values(by=sortby, ascending=ascending).head(numbers)
    print(f'Plotting first {numbers} riders in {sortby}')
    print('\nPrint the first 5 riders:')
    print(riders_info.head())

    # mark the rider on the map
    for xy in riders_info['lonlat'].to_numpy():
        x, y = literal_eval(xy)
        if x and y:
            m.plot(x, y, 'bo', markersize=1)
    plt.show()


# Points per specialty
def points():
    riders_info = pd.read_csv('riders_info_cleaned.csv')
    riders_info['total'] = riders_info['one day races'] + riders_info['GC'] + riders_info['TT'] + riders_info['sprint'] + riders_info['climber']

    min_total = 50 # flexible
    riders_info = riders_info[riders_info['total'] > min_total]

    riders_info['1day%'] = riders_info['one day races'] / riders_info['total']
    riders_info['GC%'] = riders_info['GC'] / riders_info['total']
    riders_info['TT%'] = riders_info['TT'] / riders_info['total']
    riders_info['sprint%'] = riders_info['sprint'] / riders_info['total']
    riders_info['climber%'] = riders_info['climber'] / riders_info['total']

    # sort by total points
    riders_info = riders_info.sort_values(by='total', ascending=False)
    print(riders_info.head()[['name', 'total', '1day%', 'GC%', 'TT%', 'sprint%', 'climber%']])

    # hist of total point
    riders_info.hist(column='total', bins=1000)
    plt.show()


# correlation
def correlation():
    riders_info = data_cleaning()

    corr = riders_info.corr()
    print(corr)

    plt.figure(figsize=(10, 10))
    plt.title('Correlation between (height, weight) and (PCS rank, UCI rank, All time rank)')
    sns.heatmap(corr, annot=True)
    plt.show()


if __name__ == "__main__":
    # rider = 'tadej-pogacar'
    # race(rider)

    # data_cleaning()
    # attach_birth_place_long_lat()

    # correlation()

    # hw_plot()

    # birth_place_plot()

    points()