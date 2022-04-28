import pandas as pd
import numpy as np


def race(rider):
    race_df = pd.read_csv(f'riders/{rider}.csv')
    print(race_df.head())


if __name__ == "__main__":
    rider = 'tadej-pogacar'
    race(rider)