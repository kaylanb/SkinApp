'''when this module is called, do ML and output 3 colm text file containing: prediction, answer, url'''

from numpy.random import rand
from numpy import ones, zeros, concatenate
import numpy as np
from pandas import read_csv, DataFrame, concat

# from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier

class FacesAndLimbsTrainingSet():
    def __init__(self,Food_df,Faces_df,SkinNoFaces_df):
        self.Food= Food_df.ix[:251,:].copy()
        self.People= self.Food.copy()
        for i in np.arange(0,134):
            self.People.ix[i,:]= Faces_df.ix[i,:].copy()
        cnt=0
        for i in np.arange(134,250):
            self.People.ix[i,:]= SkinNoFaces_df.ix[cnt,:].copy()
            cnt+=1

class NoLimbsTrainingSet():
    def __init__(self,Food_df,Faces_df):
        self.Food= Food_df.ix[:251,:].copy()
        self.People= Faces_df.ix[:251,:].copy()
        
class NoFacesTrainingSet():
    def __init__(self,Food_df,SkinNoFaces_df):
        self.Food= Food_df.ix[:117,:].copy()
        self.People= SkinNoFaces_df.ix[:117,:].copy()

class Team_or_Kay_Features():
    def __init__(self,Food_df,Faces_df,SkinNoFaces_df):
        self.FacesAndLimbs= FacesAndLimbsTrainingSet(Food_df,Faces_df,SkinNoFaces_df)
        self.NoLims= NoLimbsTrainingSet(Food_df,Faces_df)
        self.NoFaces= NoFacesTrainingSet(Food_df,SkinNoFaces_df)

class AddTeamCols():
    def __init__(self, Food_KayF,Faces_KayF,SkinNoFaces_KayF, 
                Food_TeamF,Faces_TeamF,SkinNoFaces_TeamF):
        self.Food= Food_KayF.copy()
        cols= Food_TeamF.columns[2:12]
        for col in cols:
            self.Food[col]= Food_TeamF[col]
        
        self.Faces= Faces_KayF.copy()
        cols= Faces_TeamF.columns[2:12]
        for col in cols:
            self.Faces[col]= Faces_TeamF[col]

        self.SkinNoFaces= SkinNoFaces_KayF.copy()
        cols= SkinNoFaces_TeamF.columns[2:12]
        for col in cols:
            self.SkinNoFaces[col]= SkinNoFaces_TeamF[col]

def totp(ans):
    return float( np.sum(ans.astype('bool')) )
def totn(ans):
    return float( np.sum(ans.astype('bool') == False) )
def tp(predict,ans):
    return float( len(np.where(predict.astype('bool') & ans.astype('bool'))[0]) )
def fp(predict,ans):
    return float( len(np.where((predict.astype('bool')==False) & ans.astype('bool'))[0]) )
def fn(predict,ans):
    return float( len(np.where((predict.astype('bool')) & (ans.astype('bool')==False))[0]) )
def precision(predict,ans):
    prec= tp(predict,ans)/(tp(predict,ans) + fp(predict,ans))
    tp_norm= tp(predict,ans)/totp(ans)
    fp_norm= fp(predict,ans)/totp(ans)
    print "tp/totp,fp/totp,precision= %f %f %f" % \
        (tp_norm,fp_norm,prec)
    return prec,tp_norm,fp_norm

def recall(predict,ans):
    rec= tp(predict,ans)/(tp(predict,ans) + fn(predict,ans))
    tp_norm= tp(predict,ans)/totp(ans)
    fn_norm= fn(predict,ans)/totn(ans)
    print "tp/totp,fn/totn,recall= %f %f %f" % \
        (tp_norm,fn_norm,rec)
    return rec,tp_norm,fn_norm

def fraction_correct(predict,ans):
    tp= float( len(np.where(predict.astype('bool') & ans.astype('bool'))[0]) )
    tn= float( len(np.where( (predict.astype('bool')==False) & (ans.astype('bool')==False) )[0]) )
    return (tp+tn)/len(ans)

def best_machine_learn_NoRandOrd(TrainX,TrainY,TestX,\
                                n_estim=100,min_samples_spl=2,scale=False):
    forest1 = RandomForestClassifier(n_estimators=n_estim, max_depth=None,
                                     min_samples_split=min_samples_spl, random_state=0,
                                    compute_importances=True)
    forest1.fit(TrainX,TrainY)
    forestOut1 = forest1.predict(TestX)
    # precision(forestOut1,TestY)
#     recall(forestOut1,TestY)
#     print sum(forestOut1 == TestY)/float(len(forestOut1))

    # forest2 = ExtraTreesClassifier(n_estimators=n_estim, max_depth=None,
#                                    min_samples_split=min_samples_spl, random_state=0,
#                                     compute_importances=True)
#     forest2.fit(TrainX,TrainY)
#     forestOut2 = forest2.predict(TestX)
#     precision(forestOut2,TestY)
#     recall(forestOut2,TestY)
#     print sum(forestOut2 == TestY)/float(len(forestOut2))

   #  forest3 = AdaBoostClassifier(n_estimators=n_estim, random_state=0)
#     forest3.fit(TrainX,TrainY)
#     forestOut3 = forest3.predict(TestX) 
#     precision(forestOut3,TestY)
#     recall(forestOut3,TestY)
#     print sum(forestOut3 == TestY)/float(len(forestOut3))

    #most important features in each classifier
    def ImpFeatures(forest,feature_list):
        df= DataFrame()
        df["importance"]=forest.feature_importances_
        df.sort(columns="importance", inplace=True,ascending=False)
        df["features"]= feature_list[df.index]
        return df
#     if importance:
#         t_df= ImpFeatures(tree,Food.columns)
#         f1_df= ImpFeatures(forest1,Food.columns)
#         f2_df= ImpFeatures(forest2,Food.columns)
#         #AdaBoostClassifier not have: compute_importances??
#         print "tree\n",t_df.head()
#         print "forest\n", f1_df.head()
#         print "forest2\n",f2_df.head()
#     return forestOut2,TestY,TestX,cTestP,cTestF,People_all,Food_all
    return forest1,forestOut1

def Train_the_RandomForest():
    Food_KayF = read_csv('csv_features/NewTraining_Food_everyones_KFeat_Toddmap.csv')
    Faces_KayF = read_csv('csv_features/NewTraining_Faces_everyones_KFeat_Toddmap.csv')
    SkinNoFaces_KayF = read_csv('csv_features/NewTraining_SkinNoFaces_everyones_KFeat_Toddmap.csv')
    Food_TeamF = read_csv('csv_features/NewTraining_Food_everyones_TeamFeat_Toddmap.csv')
    Faces_TeamF = read_csv('csv_features/NewTraining_Faces_everyones_TeamFeat_Toddmap.csv')
    SkinNoFaces_TeamF = read_csv('csv_features/NewTraining_SkinNoFaces_everyones_TeamFeat_Toddmap.csv')

    #team feature numbers for different definitions of Food,People
    team= Team_or_Kay_Features(Food_TeamF,Faces_TeamF,SkinNoFaces_TeamF)
    #kay feature numbers for different definitions of Food,People
    kay= Team_or_Kay_Features(Food_KayF,Faces_KayF,SkinNoFaces_KayF)
    #kay feature numbers + team feature number for skin maps for different definitions of Food,People
    extend= AddTeamCols(Food_KayF,Faces_KayF,SkinNoFaces_KayF, 
                        Food_TeamF,Faces_TeamF,SkinNoFaces_TeamF)
    kay_extend= Team_or_Kay_Features(extend.Food,extend.Faces,extend.SkinNoFaces)

    ##
    #make training and test sets
    Food_all = kay_extend.NoLims.Food
    People_all= kay_extend.NoLims.People
    ###
    Food=Food_all.ix[:,2:]
    People=People_all.ix[:,2:]

    sh= Food.values.shape
    max=int(sh[0]/2.)
    TrainF= Food.values[0:max,:]
    TestF = Food.values[max:,:]
    #want urls in test set to find image user selects
    TestF_URL= Food_all.URL.values[max:]

    sh= People.values.shape
    max=int(sh[0]/2.)
    TrainP= People.values[0:max,:]
    TestP = People.values[max:,:]
    #want urls in test set to find image user selects
    TestP_URL= People_all.URL.values[max:]

    TrainX = concatenate([TrainP, TrainF])
    TestX = concatenate([TestP, TestF])
    #want urls in test set to find image user selects
    TestX_URL = concatenate([TestP_URL, TestF_URL])

    TrainY = concatenate([zeros(len(TrainP)), ones(len(TrainF))])
    TestY = concatenate([zeros(len(TestP)), ones(len(TestF))])

    scale=False
    if scale:## SCALE X DATA
        from sklearn import preprocessing
        TrainX = preprocessing.scale(TrainX)
        TestX = preprocessing.scale(TestX)

    #run ML on Food vs. People train/test set of choice
    RF = RandomForestClassifier(n_estimators=100, max_depth=None,
                                         min_samples_split=2, random_state=0,
                                        compute_importances=True)
    RF.fit(TrainX,TrainY)
    return RF, TestX,TestY,TestX_URL

def output_results():
    (RF, TestX,TestY,TestX_URL)= Train_the_RandomForest()   
    Y_predict= RF.predict(TestX)
    results_df= DataFrame()
    results_df["url"]=TestX_URL
    results_df["answer"]=TestY
    results_df["predict"]=Y_predict
    import pickle
    fout = open("RandForestTrained.pickle", 'w') 
    pickle.dump(RF, fout)
    fout.close()
    fout = open("TestImageSet_predictions_answers.pickle", 'w') 
    pickle.dump(results_df, fout)
    fout.close()

# print np.where(Y_predict.astype('int') == TestY.astype('int'))[0].shape[0]/float(TestX_URL.size)
# 
# 
# url= TestX_URL[50]
# ind=np.where(TestX_URL == url)[0]
# if ind.size != 1: print "bad"
# else: 
#     print "good"
#     ind=ind[0]




