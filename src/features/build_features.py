import sys
import os
import pandas as pd
import numpy as np
import re
from datetime import datetime


riders = os.listdir('data/processed/riders')  # ['primoz-roglic.csv', 'tadej-pogacar.csv', 'alejandro-valverde.csv' ...]
riders_num = len(riders)
print(f'Total number of riders (before filter): {riders_num}') # 9039


class RiderData:
    """
    Get all rider data from processed data parquet

        min_ranks: minimum number of races in the record, to ensure the rider has enough data
    """
    def __init__(self, min_ranks:int=10):
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.master("local[*]").config("spark.driver.memory", "4g").getOrCreate()

        self.min_ranks = min_ranks
        self.riders_par = spark.read.load("data/processed/riders.parquet")

    def load_data(self, idx, min_ranks=None):
        min_ranks = min_ranks if min_ranks is not None else self.min_ranks

        riders_par = self.riders_par
        rider = riders[idx]
        rider_name = rider[:-4] # remove .csv

        rider_races = riders_par.filter(riders_par.rider == rider_name).toPandas()
        rider_races = rider_races.drop(columns=['rider'])

        rider_races.sort_values(by='date', ascending=True, inplace=True)

        rider_races['race name'] = rider_races['race name'].str.strip()
        rider_races.drop_duplicates(inplace=True)
        rider_races.dropna(inplace=True)

        # drop a few exceptions
        rider_races = rider_races[rider_races['result ranking'] != 'DNF'] # did not finish
        rider_races = rider_races[rider_races['result ranking'] != 'DNF*']
        rider_races = rider_races[rider_races['result ranking'] != 'DF']
        rider_races = rider_races[rider_races['result ranking'] != 'DNS'] # did not start
        rider_races = rider_races[rider_races['result ranking'] != 'DSQ'] # disqualied
        rider_races = rider_races[rider_races['result ranking'] != 'OTL'] # outside time limit
        rider_races['result ranking'] = rider_races['result ranking'].str.strip()

        try:
            rider_races['result ranking'] = pd.to_numeric(rider_races['result ranking'])
        except:
            def find_rank_num(s):
                nums = re.findall('\d+', s)
                if len(nums) == 1:
                    return nums[0]
                else:
                    print(f'Error: {s} at {idx}')
                    # raise Exception(f'Error: {s} at {idx}')
                    return None

            rider_races['result ranking'] = rider_races['result ranking'].apply(find_rank_num)

        rider_races.dropna(subset=['result ranking'], inplace=True)

        # at least having XX ranks in record
        if len(rider_races) < min_ranks: return
        return rider_races

    def rider_features(self, idx, min_ranks=None, date_offset=True):
        # idx: 0 to riders_num-1
        races = self.load_data(idx, min_ranks)
        if races is None: return

        features = races[['date', 'distance', 'result ranking']].to_numpy()
        if date_offset is False: return features

        for i in range(features.shape[0]): # convert string to datetime object
            features[i, 0] = self._str2dates(features[i, 0])

        return features

    def _find_max_rows(self):
        max_rows = 0
        for i in range(riders_num):
            features = self.rider_features(i)
            if features is not None:
                rows = features.shape[0]
                if rows > max_rows:
                    max_rows = rows

        print('max rows among all riders', max_rows) # 1550

    def _find_earliest_date(self):
        earliest = []
        for i in range(riders_num):
            rider = self.rider_features(i, min_ranks=1, date_offset=False)
            if rider is not None:
                earliest.append(datetime.strptime(rider[0, 0], '%Y-%m-%d'))

        print(f'The earliest date in the races is: {min(earliest)}') # 1893-06-29

    def _str2dates(self, date_str):
        offset_date = datetime(1893, 6, 28)
        date = datetime.strptime(date_str, '%Y-%m-%d')
        date = date - offset_date
        return date.days


if __name__ == "__main__":
    rider_data = RiderData()
    # print(rider_data.rider_features(1953))
    # rider_data._find_max_rows()
    rider_data._find_earliest_date()