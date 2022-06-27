# Pro cyclist races

This repo aims to collect the data from [procyclingstats](https://www.procyclingstats.com/rankings.php) and use machine learning to predict the race outcome.
Right now this repo is only capable of fetching and collecting data from [procyclingstats](https://www.procyclingstats.com/rankings.php).


| File name         | Description |
| ----------------- | ----------- |
| riders/           | The race data of riders      |
| dates.csv         | Dates of races      |
| riders_info.csv   | List of riders' info (height, DOB, ranking, etc)       |
| riders.csv        | All riders names and their link       |
| fetch.py          | Web scraper      |
| helper.csv        | Some useful helper functions      |
| read.py           | Visualize the data      |
| test.py           | For testing purposes      |
| ghosts.txt        | Link existed, but no contents      |


# Todos
- [x] fetch the rider's data, like height, points per specialty
- [ ] Visualize the data
- [ ] Setup a database for data storage (seems not necessary)
- [ ] **Build model!!**