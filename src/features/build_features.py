import os
import pandas as pd
import numpy as np


riders = os.listdir('data/processed/riders')  # ['primoz-roglic.csv', 'tadej-pogacar.csv', 'alejandro-valverde.csv' ...]
riders_num = len(riders)
print(f'Total number of riders (before filter): {riders_num}') # 9039


def load_data(idx, min_ranks=10):
    rider = riders[idx]
    rider = rider[:-4]  # remove .csv
    rider_races = pd.read_csv(f'data/processed/riders/{rider}.csv')
    rider_races = rider_races.sort_values(by='date', ascending=False)
    rider_races['race name'] = rider_races['race name'].str.strip()

    rider_races = rider_races.drop_duplicates()

    # drop DNF,OTL ranking and NaN distance
    rider_races = rider_races[rider_races['result ranking'] != 'DNF'] # did not finish
    rider_races = rider_races[rider_races['result ranking'] != 'OTL'] # outside time limit
    rider_races = rider_races[rider_races['distance'] != 'NaN']

    rider_races['result ranking'] = pd.to_numeric(rider_races['result ranking'])

    # at least having XX ranks in record
    if len(rider_races) < min_ranks:
        return
    return rider_races


# idx: 0 to riders_num-1
def rider_features(idx, min_ranks=30):
    races = load_data(idx, min_ranks=min_ranks)
    if races is None:
        return

    features = races[['date', 'distance', 'result ranking']]
    return features.to_numpy()


if __name__ == "__main__":
    rider_features(29)
