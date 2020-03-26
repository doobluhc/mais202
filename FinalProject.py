import matplotlib
from sklearn.linear_model import LogisticRegression
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm
import cPickle
from sklearn.datasets.samples_generator import make_blobs
import pylab as pl
from sklearn.utils import shuffle
#preporcess data
features_to_drop = ['Unnamed: 0', 'Game', 'Date', 'Opponent', 'OpponentPoints',
                        'Opp.FieldGoals', 'Opp.3PointShotsAttempted',
                        'Opp.FreeThrowsAttempted', 'Opp.FreeThrows.', 'Opp.OffRebounds',
                        'Opp.Assists', 'Opp.TotalFouls',
                        'Opp.FieldGoalsAttempted', 'Opp.3PointShots','FreeThrows','FieldGoals','X3PointShots'
                    ,'X3PointShotsAttempted','OpponentPoints','TotalFouls']
NBAdata = pd.read_csv('/Users/chengchen/Downloads/nba.games.stats.csv')
NBAdata = NBAdata.drop(features_to_drop, axis=1)
del NBAdata['Team']
NBAdata['Home'].replace({'Home': 1, 'Away': 0}, inplace=True)
NBAdata = shuffle(NBAdata)
results = NBAdata['WINorLOSS']

binaryResults = []
for i in range(len(results)):
    if(results[i] == 'W'):
        binaryResults.append(1)
    else:
        binaryResults.append(0)
binaryResults = np.array(binaryResults)
del NBAdata['WINorLOSS']



def logisticRegression():
    X_train, X_test, y_train, y_test = train_test_split(NBAdata, binaryResults, test_size=0.2)
    logreg = LogisticRegression(class_weight='balanced')
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print(y_pred)
    print(y_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


def SVM ():

    X_train, X_test, y_train, y_test = train_test_split(NBAdata, binaryResults, test_size=0.2)

    #train the model
    print('trainning')
    #clf = svm.SVC(kernel='linear')  # Linear Kernel
    #clf.fit(X_train, y_train)

    #save the model
    #print('save the model')
   # with open('/Users/chengchen/PycharmProjects/mais202FinalProject/clf.pkl', 'wb') as fid:
        #cPickle.dump(clf, fid)

    #load the model
    with open('/Users/chengchen/PycharmProjects/mais202FinalProject/clf.pkl', 'rb') as fid:
        clf = cPickle.load(fid)

    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(y_pred)


    #visulize the data
    X_train, y_train = make_blobs(n_samples=50, centers=2,
                                  random_state=0, cluster_std=0.60)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='autumn');
    plt.show()



SVM()