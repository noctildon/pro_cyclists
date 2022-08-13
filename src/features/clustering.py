import collections
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, DBSCAN, KMeans


def pca_analysis_on_points(save=False):
    riders_info = classify_points()
    riders_points = riders_info[['1day%', 'GC%', 'TT%', 'sprint%', 'climber%']]

    pca = PCA(n_components=3)
    pca.fit(riders_points)
    print('Variance ratio', pca.explained_variance_ratio_) # first 3 accounts for 93%

    principalComponents = pca.fit_transform(riders_points)
    riders_points_pca = pd.DataFrame(data=principalComponents, columns=['pca1', 'pca2', 'pca3'])

    # clustering = MeanShift().fit(riders_points_pca)
    # clustering = DBSCAN().fit(riders_points_pca)
    clustering = KMeans(n_clusters=3).fit(riders_points_pca)
    clustering_freq = collections.Counter(clustering.labels_)
    print('Clustering freq:', clustering_freq)

    riders_points_pca_labelled = pd.concat((riders_points_pca, pd.DataFrame(clustering.labels_, columns=['cluster'])), axis=1)

    # 3d plot
    fig = px.scatter_3d(riders_points_pca_labelled, x='pca1', y='pca2', z='pca3', color='cluster')
    fig.update_layout(title_text="3D PCA plot of riders points (variance ratio: 93% for the first 3)")
    fig.show()
    if save:
        fig.write_html("reports/figures/riders_PCA.html")


# Age, Weight, Height
def pca_analysis_on_figure(save=False):
    riders_info = pd.read_csv('data/processed/riders_info_cleaned.csv')
    this_year = datetime.date.today().year

    # Find age
    riders_info['DOB'] = pd.to_datetime(riders_info['DOB'])
    riders_info['age'] = this_year - riders_info['DOB'].dt.year
    riders_info = riders_info[['age', 'weight', 'height']]

    clustering = KMeans(n_clusters=3).fit(riders_info)
    clustering_freq = collections.Counter(clustering.labels_)
    print('Clustering freq:', clustering_freq)

    riders_info_labelled = pd.concat((riders_info, pd.DataFrame(clustering.labels_, columns=['cluster'])), axis=1)

    # 3d plot
    fig = px.scatter_3d(riders_info_labelled, x='height', y='weight', z='age', color='cluster')
    fig.update_layout(title_text="Age, weight and height of riders")
    fig.show()
    if save:
        fig.write_html("reports/figures/riders_awh.html")


# Points per specialty
def classify_points(plot=False):
    riders_info = pd.read_csv('data/processed/riders_info_cleaned.csv')
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

    # hist of total point
    if plot:
        print(riders_info.head()[['name', 'total', '1day%', 'GC%', 'TT%', 'sprint%', 'climber%']])
        riders_info.hist(column='total', bins=1000)
        plt.show()

    return riders_info


if __name__ == "__main__":
    # pca_analysis_on_points()
    pca_analysis_on_figure()