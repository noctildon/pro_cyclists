import pandas as pd
from geopy.geocoders import Nominatim


def print_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)


def get_lon_lat(address):
    geolocator = Nominatim(user_agent="myapplication")
    location = geolocator.geocode(address)
    return location.longitude, location.latitude