'''predict on same url list as Hog will predict on'''

from numpy.random import rand
from numpy import ones, zeros, concatenate
import numpy as np
from pandas import read_csv, DataFrame, concat
from pickle import dump

from sklearn.ensemble import RandomForestClassifier


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


def train_and_predict(features_df):
    TrainX= features_df.ix[0:300,2:].values
    TrainY=features_df.ix[0:300,0].values
    TestX= features_df.ix[300:,2:].values
    TestY= features_df.ix[300:,0].values
    TestUrls= features_df.ix[300:,1].values

    RF = RandomForestClassifier(n_estimators=100, max_depth=None,
                                         min_samples_split=2, random_state=0,
                                        compute_importances=True)
    RF.fit(TrainX,TrainY)
    predict= RF.predict(TestX)
    
    #save predictions to file
    results_df= DataFrame()
    results_df["url"]=TestUrls
    results_df["answer"]=TestY
    results_df["predict"]=predict
    fout = open("blob_predict_NAME.pickle", 'w') 
    dump(results_df, fout)
    fout.close()
    #save stats of run to file
    (prec,tp_norm,fp_norm)= precision(predict,TestY):
    (rec,tp_norm,fn_norm)= recall(predict,TestY)
    stats={}
    stats["prec"]= prec
    stats["rec"]= rec
    stats["fp_norm"]= fp_norm
    stats["fn_norm"]= fn_norm
    stats["frac_corr"]= fraction_correct(predict,TestY)
    fout = open("blob_predict_stats_NAME.pickle", 'w') 
    dump(stats, fout)
    fout.close()






