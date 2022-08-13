from random import gauss
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from ast import literal_eval


# rider = 'tadej-pogacar'
def race(rider):
    race_df = pd.read_csv(f'data/raw/riders/{rider}.csv')
    print(race_df.head())


# Correlation plot
def correlation():
    riders_info = pd.read_csv('data/processed/riders_info_cleaned.csv')

    corr = riders_info.corr()
    print(corr)

    plt.figure(figsize=(10, 10))
    plt.title('Correlation between (height, weight) and (PCS rank, UCI rank, All time rank)')
    sns.heatmap(corr, annot=True)
    plt.show()


# Weight vs height plot
def hw_plot():
    riders_info = pd.read_csv('data/processed/riders_info_cleaned.csv')

    # histogram of weight
    riders_info.hist(bins=50, column=['weight', 'height'])
    plt.show()

    # scatter plot
    plt.title('Riders weight and height')
    sns.scatterplot(x='weight', y='height', data=riders_info)
    plt.xlabel('Weight [kg]')
    plt.ylabel('Height [m]')
    plt.show()


def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def BMI_plot(save=False):
    riders_info = pd.read_csv('data/processed/riders_info_cleaned.csv')
    riders_info['BMI'] = riders_info['weight'] / riders_info['height'] ** 2

    # plot histogram of BMI by numpy
    plt.figure(figsize=(8, 8))
    plt.title('Histogram of BMI')
    counts, bins, bars = plt.hist(riders_info['BMI'], bins=50)
    bin_centers = (bins[1:] + bins[:-1]) / 2

    popt, pcov = curve_fit(gaussian, bin_centers, counts, bounds=((0,0), (30,15)))
    counts_fit = gaussian(bin_centers, *popt)

    print(popt)

    # plt.plot(bin_centers, counts_fit, 'r-')
    # plt.show()


    return

    plt.xlabel('BMI')
    plt.ylabel('Count')
    plt.show()

    if save:
        plt.savefig('reports/figures/BMI_hist.png')
    plt.show()


# Plot geographical birth place
# sortby = 'UCI rank', 'weight', 'dob', etc
# numbers = how many riders to plot
def birth_place_plot(sortby='UCI rank', numbers=100, ascending=True):
    riders_info = pd.read_csv('data/processed/riders_info_cleaned.csv')

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


if __name__ == "__main__":
    # birth_place_plot()
    BMI_plot()