import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
import requests
from helper import print_df
import threading
import os
from queue import Queue


url_base = 'https://www.procyclingstats.com/rankings.php'
# https://www.procyclingstats.com/rankings.php?date=2021-12-31&offset=100


# Get riders name and link (for a given date and offset) from the rankings page
# date = '2021-12-31', offset = n*100, n>=0
def get_names(date, offset=0, verbose=False):
    params = {
        'date': date,
        'offset': offset
    }
    res = requests.get(url=url_base, params=params)
    soup = BeautifulSoup(res.text, 'html.parser')
    riders_soup = soup.tbody.find_all('tr')

    riders = []
    for ride in riders_soup:
        link = ride.a.get('href')
        name = ride.a.text
        if verbose:
            print(name, link)
            print('')

        riders.append([name, link])

    return riders


# Get the results (max) number to determine the offset (for a given date)
# date = '2021-12-31'
def get_results_number(date):
    params = {
        'date': date
    }
    res = requests.get(url=url_base, params=params)
    soup = BeautifulSoup(res.text, 'html.parser')
    num = soup.b.text.split('/')[-1]
    return int(num)


# Get the available dates
def get_dates():
    res = requests.get(url=url_base)
    soup = BeautifulSoup(res.text, 'html.parser')
    s = soup.find_all('select', {'name':'date'})[0]
    options = s.find_all('option')
    dates = [option.text for option in options]

    return dates


# Update all riders in rankings.php and save to riders.csv
# Takes around 30 mins
def update_all_riders(verbose=False):
    riders = []
    dates = get_dates()
    for date in dates:
        maxnum = get_results_number(date)
        for offset in range(0, maxnum//100 * 100 + 100, 100):
            riders += get_names(date=date, offset=offset, verbose=verbose)

    riders_df = pd.DataFrame(riders, columns=['Rider name', 'The link'])
    riders_df = riders_df.drop_duplicates()
    riders_df.to_csv('riders.csv', index=False)


# much faster than update_all_riders
def update_all_riders_concurrent(verbose=False):
    riders = []
    threads = []

    q = Queue()
    def job(date, offset, q):
        riders_2add = get_names(date=date, offset=offset, verbose=verbose)
        q.put(riders_2add)

    dates = get_dates()
    for date in dates:
        maxnum = get_results_number(date)
        offsets = range(0, maxnum//100 * 100 + 100, 100)
        for offset in offsets:
            t = threading.Thread(target=job, args=(date, offset,q))
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    for _ in range(len(threads)):
        riders.append(q.get())

    riders_df = pd.DataFrame(riders, columns=['Rider name', 'The link'])
    riders_df = riders_df.drop_duplicates()
    riders_df.to_csv('riders.csv', index=False)


# Rmove the duplicate riders in existing riders.csv
def remove_duplicate_riders():
    riders = pd.read_csv('riders.csv')
    riders = riders.drop_duplicates()
    riders.to_csv('riders.csv', index=False)


# Update dates
def update_dates():
    dates = get_dates()
    dates = pd.DataFrame(dates, columns=['date'])
    dates_pre = pd.read_csv('dates.csv')

    if dates.equals(dates_pre):
        print('nothing changed. no need to update')
    else:
        dates.to_csv('dates.csv', index=False)
        print('dates.csv updated')


# Get race for a ride in a year (year_link)
# year_link = 'https://www.procyclingstats.com/rider/tadej-pogacar/2019
def get_race(year_link, verbose=False):
    races = []

    res = requests.get(url=year_link)
    soup = BeautifulSoup(res.text, 'html.parser')
    table = soup.find('tbody').find_all('tr')
    for row_ in table:
        row = row_.find_all('td')

        date = row[0].text
        result_ranking = row[1].text
        gc = row[2].text
        race_name = row[4].text
        distance = row[5].text
        pcs_pt = row[6].text
        uci_pt = row[7].text
        race = [date, result_ranking, gc, race_name, distance, pcs_pt, uci_pt]
        races.append(race)

    races_df = pd.DataFrame(races, columns=['date', 'result ranking', 'general classification', 'race name', 'distance', 'PCS point', 'UCI point'])
    if verbose:
        print_df(races_df)
    return races_df


# Get all the race years of a rider
# link = 'https://www.procyclingstats.com/rider/wout-van-aert'
def get_race_years(link):
    years = []
    res = requests.get(url=link)

    soup = BeautifulSoup(res.text, 'html.parser')
    s = soup.find_all('span',string='more')
    lis = s[0].parent.find_all('li')
    for li in lis:
        years.append(int(li.text))

    return years


# Get all race of a rider
# rider_link = 'rider/tadej-pogacar'
def get_all_race(rider_link, this_year_only=False):
    url = 'https://www.procyclingstats.com/'
    url += rider_link

    years = get_race_years(url)
    races = []
    for year in years:
        race = get_race(f'{url}/{year}')
        races.append(race)
        if this_year_only:
            break

    return pd.concat(races)


def update_all_riders_races(verbose=False):
    riders_df = pd.read_csv('riders.csv')

    for i in range(len(riders_df)):
        link = riders_df.loc[i, "The link"]        # 'rider/tadej-pogacar'
        name = link[6:]                            # 'tadej-pogacar'
        textname = riders_df.loc[i, "Rider name"]  # 'Pogačar Tadej'

        races = get_all_race(link)
        if verbose:
            print_df(races)

        races.to_csv(f'riders/{name}.csv', index=False)
        print(f'{textname}\'s races saved')


def update_all_riders_races_concurrent(renew=False, this_year_only=False):
    riders_df = pd.read_csv('riders.csv')

    def get_all_race_concurrent(link, name, textname):
        races = get_all_race(link, this_year_only=this_year_only)
        races.to_csv(f'riders/{name}.csv', index=False)
        print(f'{textname}\'s races saved')

    # ['primoz-roglic.csv', 'tadej-pogacar.csv', 'alejandro-valverde.csv' ...]
    downloaded = os.listdir('riders')
    undownloaded = []

    # num = 10 # for testing purposes
    num = len(riders_df)
    for i in range(num):
        link = riders_df.loc[i, "The link"]        # 'rider/tadej-pogacar'
        name = link[6:]                            # 'tadej-pogacar'
        textname = riders_df.loc[i, "Rider name"]  # 'Pogačar Tadej'

        if renew:
            undownloaded.append([link,name,textname])
        else:
            if f'{name}.csv' not in downloaded:
                undownloaded.append([link,name,textname])

    threads = [threading.Thread(target=get_all_race_concurrent, args=(d[0],d[1],d[2])) for d in undownloaded]
    for t in threads:
        t.start()


if __name__ == "__main__":
    # update_dates() # the website update the latest date every day

    # update_all_riders_concurrent()
    update_all_riders_races_concurrent()
