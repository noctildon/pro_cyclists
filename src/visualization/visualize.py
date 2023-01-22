import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from ast import literal_eval

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)


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


def BMI_plot(save=False):
    riders_info = pd.read_csv('data/processed/riders_info_cleaned.csv')
    riders_info['BMI'] = riders_info['weight'] / riders_info['height'] ** 2

    # plot histogram of BMI by numpy
    plt.figure(figsize=(8, 8))
    plt.title('Histogram of BMI')
    counts, bins, bars = plt.hist(riders_info['BMI'], bins=50)
    bin_centers = (bins[1:] + bins[:-1]) / 2

    # gaussian fit
    def gaussian(x, mu, sigma, c):
        return c * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    maxidx = np.argmax(counts)
    mean = bin_centers[maxidx]
    std = np.std(counts)
    popt, pcov = curve_fit(gaussian, bin_centers, counts, p0=(mean, std, 1), bounds=(0, np.inf))
    counts_fit = gaussian(bin_centers, *popt)
    print(popt)

    plt.plot(bin_centers, counts_fit, 'r-', label=rf'$\mu={popt[0]:.2f}, \sigma={popt[1]:.2f}, c={popt[2]:.2f}$')
    plt.xlabel('BMI')
    plt.ylabel('Count')
    plt.title('Histogram of BMI with ' + r'$f(x)=c\,\,  exp(-0.5 [ \frac{x-\mu}{\sigma}]^2)$')
    plt.legend()
    plt.tight_layout()

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


def plot_results():
    # id, Avg, 1D, XGBoost, LightGBM, NN, LSTM
    results_data = pd.read_csv('models/indep_models_results.txt', sep=' ')
    print(results_data.info())

    bins = np.linspace(0, 0.5, 50)
    plt.hist(results_data['Avg'], bins=bins, label='Avg', linewidth=2, histtype='step')
    plt.hist(results_data['1D'], bins=bins, label='1D', linewidth=2, histtype='step')
    plt.hist(results_data['XGBoost'], bins=bins, label='XGBoost', linewidth=2, histtype='step')
    plt.hist(results_data['LightGBM'], bins=bins, label='LightGBM', linewidth=2, histtype='step')
    plt.hist(results_data['NN'], bins=bins, label='NN', linewidth=2, histtype='step')
    plt.hist(results_data['LSTM'], bins=bins, label='LSTM', linewidth=2, histtype='step')
    plt.legend(loc='upper right')
    plt.xlabel('MAE', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.title('Histogram of MAE', fontsize=20)
    plt.show()


def stats_results():
    # Avg, 1D, XGBoost, LightGBM, NN, LSTM
    results_data = pd.read_csv('models/indep_models_results.txt', sep=' ').drop(columns=['id'])
    means = results_data.mean()
    stds = results_data.std()

    plt.figure(figsize=(8, 6))
    plt.errorbar(x=means.index, y=means, yerr=stds, fmt='o', capsize=10, elinewidth=3)
    plt.xlabel('Model', fontsize=16)
    plt.ylabel('MAE', fontsize=16)
    plt.title('Mean and std of MAE', fontsize=20)
    plt.show()


if __name__ == "__main__":
    # birth_place_plot()
    BMI_plot()