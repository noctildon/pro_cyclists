from fetch import *
from termcolor import colored

textname = 'john'
print(colored(f'{textname}\'s races saved!!', 'green'))

# year_link = 'https://www.procyclingstats.com/rider/wout-van-aert/2020'
# races = get_race(year_link)
# print(races)

# x = int(year_link.split('/')[-1])
# print(x)
# print(type(x))