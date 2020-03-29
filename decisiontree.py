
# data scraping
from basketball_reference_web_scraper import client
# client.season_schedule(season_end_year=2017,output_file_path='/Users/chengchen/PycharmProjects/DecisionTreeMAIS202/2016-2017.csv',output_type= 'OutputType.CSV')
# client.season_schedule(season_end_year=2018,output_file_path='/Users/chengchen/PycharmProjects/DecisionTreeMAIS202/2017-2018.csv',output_type= 'OutputType.CSV')
# client.season_schedule(season_end_year=2019,output_file_path='/Users/chengchen/PycharmProjects/DecisionTreeMAIS202/2018-2019.csv',output_type= 'OutputType.CSV')

import pandas as pd
import numpy as np
dataset = pd.read_csv('/Users/chengchen/PycharmProjects/DecisionTreeMAIS202/2018-2019.csv' ,parse_dates=['start_time'])
dataset.columns = ['Date', 'Visitor Team', 'Visitor Score', 'Home Team', 'Home Score']
dataset['Home Win'] = dataset['Home Score'] > dataset['Visitor Score']
y_true = dataset["Home Win"].values
from sklearn.metrics import f1_score, make_scorer, classification_report
scorer = make_scorer(f1_score, pos_label = None, average = 'weighted')

# determine if teams won their last game
dataset['Home Last Win'] = False
dataset['Visitor Last Win'] = False
from collections import defaultdict
won_last = defaultdict(int)
for index, row in dataset.iterrows():
    home_team = row['Home Team']
    visitor_team = row['Visitor Team']
    row['Home Last Win'] = won_last[home_team]
    row['Visitor Last Win'] = won_last[visitor_team]
    won_last[home_team] = row['Home Win']
    won_last[visitor_team] = not row['Home Win']

# determine the winning streak for each team
dataset["Home Win Streak"] = 0
dataset["Visitor Win Streak"] = 0
win_streak = defaultdict(int)
for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    row["Home Win Streak"] = win_streak[home_team]
    row["Visitor Win Streak"] = win_streak[visitor_team]
    dataset.loc[index] = row
    # Set current win streak
    if row["Home Win"]:
        win_streak[home_team] += 1
        win_streak[visitor_team] = 0
    else:
        win_streak[home_team] = 0
        win_streak[visitor_team] += 1

# determine who won their last match up
last_game_winner = defaultdict(int)
dataset["Home Team Won Last"] = 0
for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    teams = tuple(sorted([home_team, visitor_team]))
    row["Home Team Won Last"] = 1 if last_game_winner[teams] == row["Home Team"] else 0
    dataset.loc[index] = row
    # update for this time
    winner = row["Home Team"] if row["Home Win"] else row["Visitor Team"]
    last_game_winner[teams] = winner

# taking previous total wins into consideration
dataset_previous = pd.read_csv('/Users/chengchen/PycharmProjects/DecisionTreeMAIS202/2017-2018.csv'
                               ,parse_dates=['start_time'])
dataset_previous.columns = ['Date', 'Visitor Team', 'Visitor Score', 'Home Team', 'Home Score']
dataset_previous['Home Win'] = dataset_previous['Home Score'] > dataset_previous['Visitor Score']
previous_wins = defaultdict(int)
for index, row in dataset_previous.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    if row["Home Win"]:
        previous_wins[home_team] += 1
    else:
        previous_wins[visitor_team] += 1
dataset["Home Team Ranks Higher"] = 0
for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    row["Home Team Ranks Higher"] = int(previous_wins[home_team] > previous_wins[visitor_team])
    dataset.loc[index] = row



from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state = 14)
from sklearn.model_selection import cross_val_score
X_train = dataset[["Home Last Win", "Home Win Streak" ,'Home Team Won Last' ,'Home Team Ranks Higher']].values
scores = cross_val_score(dtc, X_train, y_true, scoring = scorer)

# Print results
print('NBA outcomes prediction for season 2018-2019')
print('F1 Accuracy rate: {0:.4f}%'.format(np.mean(scores) * 100))