'''outputs Stats for ET,SVC method for NTrial (default 10) to file using pickle'''

from numpy.random import rand
from numpy import ones, zeros, concatenate
from pandas import read_csv, DataFrame
from pandas import concat as pd_concat
import matplotlib.pyplot as plt
import pickle

from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm


class TpAndFn():
    def __init__(self,NTrials):
        self.TP= zeros(NTrials) -1
        self.TN= self.TP.copy()
        self.FP= self.TP.copy()
        self.FN= self.TP.copy()
    def CalcStats(self,P,N):
        if self.TP.astype('int')[0] == -1:
            raise ValueError
        else:
            self.precision= self.TP/(self.TP+self.FP)
            self.recall= self.TP/(self.TP + self.FN)
            self.TP_norm= self.TP/P
            self.FP_norm= self.FP/P
            self.FN_norm= self.FN/N
    def PrintStats(self):
        try:
            print "Mean precision, recall, FP_norm, FN_norm = %f %f %f %f" %\
                (self.precision.mean(), self.recall.mean(),\
                 self.FP_norm.mean(),self.FN_norm.mean())
        except AttributeError:
            print "No Data: PrimaryMLStats() has not been called yet"
            raise AttributeError
        
        
class CollectStats():
    def __init__(self,NTrials):
        self.P= zeros(NTrials) -1
        self.N= self.P.copy()
        self.ET= TpAndFn(NTrials)
        self.SVC= TpAndFn(NTrials)
    def CalcStats(self):
        self.ET.CalcStats(self.P,self.N)
        self.SVC.CalcStats(self.P,self.N)
    def PrintStats(self):
        print "ET Stats:"
        self.ET.PrintStats()
        print "SVC Stats:"
        self.SVC.PrintStats()
        
def doML_NTrials_Times(Food_df,People_df,NTrials):
    stats= CollectStats(NTrials)
    for n in range(0,NTrials):
        print "n= %d" % n
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

        stats.P[n] = len(People_df[cTestP])
        stats.N[n] = len(Food_df[cTestF])

        forest2 = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0)
        forest2.fit(TrainX,TrainY)
        forestOut2 = forest2.predict(TestX)                              
        stats.ET.TP[n] = sum(forestOut2[0:stats.P[n]] == TestY[0:stats.P[n]])
        stats.ET.TN[n] = sum(forestOut2[stats.P[n]+1:] == TestY[stats.P[n]+1:])
        stats.ET.FP[n] = stats.N[n] - stats.ET.TN[n]
        stats.ET.FN[n] = stats.P[n] - stats.ET.TP[n]

        clf2 = svm.LinearSVC()
        clf2.fit(TrainX,TrainY)
        clfOut2 = clf2.predict(TestX)
        stats.SVC.TP[n] = sum(clfOut2[0:stats.P[n]] == TestY[0:stats.P[n]])
        stats.SVC.TN[n] = sum(clfOut2[stats.P[n]+1:] == TestY[stats.P[n]+1:])
        stats.SVC.FP[n] = stats.N[n] - stats.SVC.TN[n]
        stats.SVC.FN[n] = stats.P[n] - stats.SVC.TP[n]
    return stats

## Testing
Food_df = read_csv('csv_features/hog_features_9_NewTraining_Food_everyones.csv')
People_df = read_csv('csv_features/hog_features_9_NewTraining_Faces_everyones.csv')

#compute average stats from NTrials
NTrials=10
stats = doML_NTrials_Times(Food_df,People_df,NTrials)
stats.CalcStats()
# stats.PrintStats()
#save stats to file for easy WebApp look up 
et_stats={"prec":stats.ET.precision.mean(),'recall':stats.ET.recall.mean(),\
         "fp_norm":stats.ET.FP_norm.mean(),"fn_norm":stats.ET.FN_norm.mean()}
svc_stats={"prec":stats.SVC.precision.mean(),'recall':stats.SVC.recall.mean(),\
         "fp_norm":stats.SVC.FP_norm.mean(),"fn_norm":stats.SVC.FN_norm.mean()}
stats_dict= {}
stats_dict["ET"]=et_stats
stats_dict["SVC"]=svc_stats
name= "hog_stats_%d.pickle" % NTrials
fout = open(name, 'w') 
pickle.dump(stats_dict, fout)
fout.close()

# fin=open('hog_stats_10.pickle',"r")
# test=pickle.load(fin)
# fin.close()