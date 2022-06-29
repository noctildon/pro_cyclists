import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from helper import *


# rider = 'tadej-pogacar'
def race(rider):
    race_df = pd.read_csv(f'riders/{rider}.csv')
    print(race_df.head())


def data_cleaning(verbose=False):
    riders_info = pd.read_csv('riders_info.csv')

    # -1 = missing value, and convert to integer
    riders_info['PCS rank'] = riders_info['PCS rank'].fillna(-1).astype(int)
    riders_info['UCI rank'] = riders_info['UCI rank'].fillna(-1).astype(int)
    riders_info['All time rank'] = riders_info['All time rank'].fillna(-1).astype(int)

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
        time.sleep(1.5)

    print('\n\nDone searching....')
    place_map_df = pd.DataFrame(place_map, columns=['address', 'lon', 'lat'])
    place_map_df.to_csv('place_map.csv', index=False)

    print('\nThese are the failed places:')
    print(place_map_fail)
    with open('place_map_fail.txt', 'w') as f:
        for place in place_map_fail:
            f.write(f'{place}\n')


# plot geographical birth place
def birth_place_plot():
    plt.figure(figsize=(8, 8))
    m = Basemap(projection='cyl', resolution=None, llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180,lat_0=0, lon_0=0)
    # m = Basemap(projection='moll', resolution=None, lat_0=0, lon_0=0) # for mollweide projection
    m.etopo(scale=0.5, alpha=0.5)

    places = pd.read_csv('place_map.csv')
    for x,y in zip(places['lon'].to_numpy(), places['lat'].to_numpy()):
        m.plot(x, y, 'bo', markersize=5)
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

    # correlation()

    # hw_plot()

    birth_place_plot()
