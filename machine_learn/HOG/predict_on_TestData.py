from numpy.random import rand
from numpy import ones, zeros, concatenate
from pandas import read_csv, DataFrame
from pandas import concat as pd_concat
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

def predict_TestData(Food_df,People_df):
    cTrainF = rand(len(Food_df)) > .5
    cTestF = ~cTrainF
    cTrainP = rand(len(People_df)) > .5
    cTestP = ~cTrainP

    TrainX_df = pd_concat([People_df[cTrainP], Food_df[cTrainF]],axis=0)
    TestX_df = pd_concat([People_df[cTestP], Food_df[cTestF]],axis=0)

    TrainX= TrainX_df.ix[:,2:].values
    TestX= TestX_df.ix[:,2:].values
    TrainY = concatenate([ones(len(People_df[cTrainP])), zeros(len(Food_df[cTrainF]))])
    TestY = concatenate([ones(len(People_df[cTestP])), zeros(len(Food_df[cTestF]))])

    ET_classifier = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0)
    ET_classifier.fit(TrainX,TrainY)
    ET_prediction = ET_classifier.predict(TestX) 

    LinSVC_classifier = svm.LinearSVC()
    LinSVC_classifier.fit(TrainX,TrainY)
    LinSVC_predict = LinSVC_classifier.predict(TestX)

    a=DataFrame()
    a["url"]=TestX_df.urls.values
    a["answer"]=TestY
    a["ET_predict"]=ET_prediction
    a["LinSVC_predict"]=LinSVC_predict
    a.to_csv("prediction_for_TestData.csv")

Food_df = read_csv('csv_features/hog_features_9_NewTraining_Food_everyones.csv')
People_df = read_csv('csv_features/hog_features_9_NewTraining_Faces_everyones.csv')

predict_TestData(Food_df,People_df)


