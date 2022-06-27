from turtle import down
from matplotlib.pyplot import clim
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
import requests
import scipy as sp
from helper import print_df
import threading
import os
import sys
from queue import Queue
from termcolor import colored
from datetime import datetime, date, timedelta



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


# Get riders catalog in rankings.php and save to riders.csv
# Takes around 30 mins
def get_riders_catalog(verbose=False):
    riders = []
    dates = get_dates()
    for date in dates:
        maxnum = get_results_number(date)
        for offset in range(0, maxnum//100 * 100 + 100, 100):
            riders += get_names(date=date, offset=offset, verbose=verbose)

    riders_df = pd.DataFrame(riders, columns=['Rider name', 'The link'])
    riders_df = riders_df.drop_duplicates()
    riders_df.to_csv('riders.csv', index=False)


def get_riders_catalog_concurrent(verbose=False):
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
# year_link = 'https://www.procyclingstats.com/rider/tadej-pogacar/2019'
def get_race(year_link, verbose=False):
    year = int(year_link.split('/')[-1])
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
    races_df['year'] = year
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

    try:
        lis = s[0].parent.find_all('li')
        for li in lis:
            years.append(int(li.text))
        return years
    except IndexError:
        sys.exit(f'Index error')


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


def get_all_riders_races(verbose=False, this_year_only=False):
    riders_df = pd.read_csv('riders.csv')

    for i in range(len(riders_df)):
        link = riders_df.loc[i, "The link"]        # 'rider/tadej-pogacar'
        name = link[6:]                            # 'tadej-pogacar'
        textname = riders_df.loc[i, "Rider name"]  # 'Pogačar Tadej'

        races = get_all_race(link, this_year_only=this_year_only)
        if verbose:
            print_df(races)

        if this_year_only:
            races_old = pd.read_csv(f'riders/{name}.csv', dtype={'general classification':object, 'PCS point':object, 'UCI point':object})
            races = pd.concat([races, races_old]).drop_duplicates().reset_index(drop=True)

        races.to_csv(f'riders/{name}.csv', index=False)
        print(colored(f'{textname}\'s races saved!!', 'green'))
        return


def get_all_riders_races_concurrent(mode='old', this_year_only=False):
    riders_df = pd.read_csv('riders.csv')
    with open('./ghosts.txt') as f:
        ghosts = [line.strip()[6:] + '.csv' for line in f.readlines()]

    def get_all_race_concurrent(link, name, textname):
        races = get_all_race(link, this_year_only=this_year_only)
        if this_year_only:
            races_old = pd.read_csv(f'riders/{name}.csv', dtype={'general classification':object, 'PCS point':object, 'UCI point':object})
            races = pd.concat([races, races_old]).drop_duplicates().reset_index(drop=True)
        races.to_csv(f'riders/{name}.csv', index=False)
        print(colored(f'{textname}\'s races saved!!', 'green'))


    downloaded = os.listdir('riders') # ['primoz-roglic.csv', 'tadej-pogacar.csv', 'alejandro-valverde.csv' ...]
    todownload = []

    # Rmove the ghost riders
    for ghost in ghosts:
        if ghost in downloaded:
            downloaded.remove(ghost)
            print(f'{ghost} is removed from the download queue')

    num = len(riders_df)
    for i in range(num):
        link = riders_df.loc[i, "The link"]        # 'rider/tadej-pogacar'
        name = link[6:]                            # 'tadej-pogacar'
        textname = riders_df.loc[i, "Rider name"]  # 'Pogačar Tadej'
        csvname = f'{name}.csv'
        filepath = f'riders/{csvname}'

        if mode == 'all':
            # download all riders in riders.csv, even if they are already downloaded
            todownload.append([link,name,textname])

        elif mode == 'new':
            # download only new riders (that are not downloaded yet)
            if csvname not in downloaded:
                todownload.append([link,name,textname])

        elif mode == 'old':
            # download and update only old riders (that are already downloaded before the date)
            update_days = 7  # update the files modified before 7 days ago
            this_year_only = True
            if csvname in downloaded:
                mtime = os.path.getmtime(filepath)
                if mtime < datetime.timestamp(datetime.now() - timedelta(days=update_days)):
                    todownload.append([link,name,textname])
        else:
            raise ValueError('mode must be one of "all", "new" or "old"')


    print(f'\n{len(todownload)} riders to download...')
    print('................................................')
    threads = [threading.Thread(target=get_all_race_concurrent, args=(d[0],d[1],d[2])) for d in todownload]
    for t in threads:
        t.start()


# Get the rider information (eg. dob, height, weight, etc)
# link = 'https://www.procyclingstats.com/rider/wout-van-aert'
def get_rider_info(link, verbose=False):
    if verbose:
        print('')
        print(link)
        print('Fetching rider info...')

    res = requests.get(url=link)
    soup = BeautifulSoup(res.text, 'html.parser')
    textname = soup.find('h1').text         # 'Pogačar Tadej'
    textname = " ".join(textname.split())   # revmoe the redundant spaces
    rider_link = link[32:]                  # 'rider/tadej-pogacar'

    s = soup.find('div', {'class': 'rdr-info-cont'})

    try:
        dob_elements = s.find('b', string='Date of birth:')
        dob_DD = int(dob_elements.next_sibling)              # day 12
        dob_MMYY = list(dob_elements.next_siblings)[2]
        dob_MM = dob_MMYY.split(' ')[1]                      # month September
        dob_YYYY = int(dob_MMYY.split(' ')[2])               # year 1999
        dob_str = f'{dob_MM} {dob_DD} {dob_YYYY}'
        dob = datetime.strptime(dob_str ,'%B %d %Y').date()  # 1998-08-11
    except:
        dob = None

    try:
        weight = int(s.find('b', string='Weight:').next_sibling.split(' ')[1])    # kg
    except:
        weight = None

    try:
        height = float(s.find('b', string='Height:').next_sibling.split(' ')[1])  # meter
    except:
        height = None

    try:
        place = list(s.find('b', string='Place of birth:').next_siblings)[1].text
    except:
        place = None

    specialties = s.find('h4', string='Points per specialty').parent.find('ul').find_all('div', {'class': 'pnt'})
    one_day_races, gc, time_trial, sprint, climber = [int(s.text) for s in specialties]

    try:
        pcs_rank = int(list(s.find('a', {'href': 'rankings/me/individual'}).next_elements)[1].text)
    except:
        pcs_rank = None

    try:
        uci_rank = int(list(s.find('a', {'href': 'rankings/me/uci-individual'}).next_elements)[1].text)
    except:
        uci_rank = None

    try:
        all_time_rank = int(list(s.find('a', {'href': 'rankings/me/all-time'}).next_elements)[1].text)
    except:
        all_time_rank = None

    rider_info_df = pd.DataFrame(
        [[textname, dob, weight, height, place, one_day_races, gc, time_trial, sprint, climber, pcs_rank, uci_rank, all_time_rank, rider_link]],
        columns=['name', 'DOB', 'weight [kg]', 'height [m]', 'birth place', 'one day races', 'GC', 'TT', 'sprint', 'climber', 'PCS rank', 'UCI rank', 'All time rank', 'link']
    )

    if verbose:
        print_df(rider_info_df)

    return rider_info_df


# number = how many riders to download. -1 means all
def get_all_riders_info(number=-1, renew=False, verbose=False):
    # ['primoz-roglic.csv', 'tadej-pogacar.csv', 'alejandro-valverde.csv' ...]
    all_riders_csv_list = os.listdir('riders')

    with open('./ghosts.txt') as f:
        ghosts = [line.strip() for line in f.readlines()]
        ghosts_set = set(ghosts)

    downloaded_df = pd.read_csv('riders_info.csv')
    downloaded_set = set(downloaded_df['link'].to_numpy())
    all_riders_set = set(['rider/'+rider[:-4] for rider in all_riders_csv_list])
    undownloaded = all_riders_set - downloaded_set - ghosts_set

    if renew:
        to_download_set = all_riders_set
    else:
        to_download_set = undownloaded

    i = 0
    for link in to_download_set:
        if i > number and number != -1:
            break
        link = f'https://www.procyclingstats.com/{link}'
        new_rider_info_df = get_rider_info(link, verbose=verbose)
        new_rider_info_df.to_csv('riders_info.csv', mode='a', index=False, header=False)
        i += 1



if __name__ == "__main__":
    # update_dates() # the website update the latest date every day

    # get_riders_catalog_concurrent()

    get_all_riders_races_concurrent()

    # get_all_riders_info(verbose=True)

    # get_all_riders_races(this_year_only=True)
