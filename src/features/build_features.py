import os
import pandas as pd
import numpy as np
import re


riders = os.listdir('data/processed/riders')  # ['primoz-roglic.csv', 'tadej-pogacar.csv', 'alejandro-valverde.csv' ...]
riders_num = len(riders)
print(f'Total number of riders (before filter): {riders_num}') # 9039


def load_data(idx, min_ranks=10):
    rider = riders[idx]
    rider = rider[:-4]  # remove .csv
    rider_races = pd.read_csv(f'data/processed/riders/{rider}.csv', dtype={'result ranking': str})
    rider_races = rider_races.sort_values(by='date', ascending=True)
    rider_races['race name'] = rider_races['race name'].str.strip()

    rider_races = rider_races.drop_duplicates()

    # drop a few exceptions
    rider_races = rider_races[rider_races['result ranking'] != 'DNF'] # did not finish
    rider_races = rider_races[rider_races['result ranking'] != 'DNF*']
    rider_races = rider_races[rider_races['result ranking'] != 'DF']
    rider_races = rider_races[rider_races['result ranking'] != 'DNS'] # did not start
    rider_races = rider_races[rider_races['result ranking'] != 'DSQ'] # disqualied
    rider_races = rider_races[rider_races['result ranking'] != 'OTL'] # outside time limit
    rider_races = rider_races[rider_races['distance'] != 'NaN']

    rider_races['result ranking'] = rider_races['result ranking'].str.strip()
    try:
        rider_races['result ranking'] = pd.to_numeric(rider_races['result ranking'])
    except:
        def find_rank_num(s):
            nums = re.findall('\d+', s)
            if len(nums) == 1:
                return nums[0]
            else:
                raise Exception(f'Error: {s} at {idx}')

        rider_races['result ranking'] = rider_races['result ranking'].apply(find_rank_num)

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


def find_max_rows():
    max_rows = 0
    for i in range(riders_num):
        features = rider_features(i)
        if features is not None:
            rows = features.shape[0]
            if rows > max_rows:
                max_rows = rows

    print('max rows among all riders', max_rows) # 1550


if __name__ == "__main__":
    rider_features(29)
