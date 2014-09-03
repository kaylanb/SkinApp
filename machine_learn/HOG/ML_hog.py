from numpy.random import rand
from numpy import ones, zeros, concatenate
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
from numpy import savetxt

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

Food = read_csv('Food_All_3.csv')
People = read_csv('People_All_3.csv')

print "Data loaded"

P=zeros(100)
N=zeros(100)

TP2=zeros(100)
TN2=zeros(100)
FP2=zeros(100)
FN2=zeros(100)

TP6=zeros(100)
TN6=zeros(100)
FP6=zeros(100)
FN6=zeros(100)

combined_array = zeros((10, 100))

for n in range(0,100):
    print n
    
    cTrainF = rand(len(Food)) > .3
    cTestF = ~cTrainF
    cTrainP = rand(len(People)) > .3
    cTestP = ~cTrainP

    TrainX = concatenate([People[cTrainP], Food[cTrainF]])
    TestX = concatenate([People[cTestP], Food[cTestF]])
    TrainY = concatenate([ones(len(People[cTrainP])), zeros(len(Food[cTrainF]))])
    TestY = concatenate([ones(len(People[cTestP])), zeros(len(Food[cTestF]))])

    print "Test/train selected"

    P[n] = len(People[cTestP])
    N[n] = len(Food[cTestF])

    forest2 = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0)
    forest2.fit(TrainX,TrainY)
    forestOut2 = forest2.predict(TestX)                             
    TP2[n] = sum(forestOut2[0:P[n]] == TestY[0:P[n]])
    TN2[n] = sum(forestOut2[P[n]+1:] == TestY[P[n]+1:])
    FP2[n] = N[n] - TN2[n]
    FN2[n] = P[n] - TP2[n]

    print "ET classifier completed"

    clf2 = svm.LinearSVC()
    clf2.fit(TrainX,TrainY)
    clfOut2 = clf2.predict(TestX)
    TP6[n] = sum(clfOut2[0:P[n]] == TestY[0:P[n]])
    TN6[n] = sum(clfOut2[P[n]+1:] == TestY[P[n]+1:])
    FP6[n] = N[n] - TN6[n]
    FN6[n] = P[n] - TP6[n]

    print "SVM classifier completed"

combined_array = [P, N, TP2, TN2, FP2, FN2, TP6, TN6, FP6, FN6]
savetxt("ML_training_3.csv", combined_array, delimiter=",")
