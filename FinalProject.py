import pandas as pd
import io
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
def logisticRegression():
    data = pd.read_csv('/Users/chengchen/Downloads/nba.games.stats.csv')
    #shuffle the data
    data = data.reindex(np.random.permutation(data.index))
    #drop useless data
    features_to_drop = ['Unnamed: 0', 'Game', 'Date', 'Opponent', 'OpponentPoints',
                        'Opp.FieldGoals', 'Opp.3PointShotsAttempted', 'Opp.3PointShots.', 'Opp.FreeThrows',
                        'Opp.FreeThrowsAttempted', 'Opp.FreeThrows.', 'Opp.OffRebounds', 'Opp.TotalRebounds',
                        'Opp.Assists', 'Opp.Steals', 'Opp.Blocks', 'Opp.Turnovers', 'Opp.TotalFouls',
                        'Opp.FieldGoalsAttempted', 'Opp.FieldGoals.', 'Opp.3PointShots']
    data = data.drop(features_to_drop, axis=1)

    #Assists
    X_train_assist, X_test_assist,y_train_assist,y_test_assist = train_test_split(data[['Assists']],data.WINorLOSS,test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train_assist,y_train_assist)
    assistCount = []
    for outcome in model.predict(X_test_assist):
        if (outcome == 'W'):
            assistCount.append(1)
        else:
            assistCount.append(0)


    #team points
    X_train_TeamPoints, X_test_TeamPoints, y_train_TeamPoints, y_test_TeamPoints = train_test_split(data[['TeamPoints']], data.WINorLOSS,
                                                                                    test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train_TeamPoints, y_train_TeamPoints)
    teamPointsCount = []
    for outcome in model.predict(X_test_TeamPoints):
        if (outcome == 'W'):
            teamPointsCount.append(1)
        else:
            teamPointsCount.append(0)





    averagePercentage = (model.score(X_test_assist,y_test_assist)
                         + model.score(X_test_TeamPoints, y_test_TeamPoints))/2

    print(averagePercentage)

logisticRegression()



