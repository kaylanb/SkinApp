'''predict on same url list as Hog will predict on'''

from numpy.random import rand
from numpy import ones, zeros, concatenate
import numpy as np
from pandas import read_csv, DataFrame, concat
from pickle import dump

from sklearn.ensemble import RandomForestClassifier
from pickle import load,dump


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


def train_and_predict(feat_df,url_df,predict_save_name,stats_save_name):
    TrainX= feat_df.values[0:300,:]
    TrainY=url_df.answer.values[0:300]
    TestX= feat_df.values[300:,:]
    TestY= url_df.answer.values[300:]
    TestUrls= url_df.URL.values[300:]

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
    fout = open(predict_save_name, 'w') 
    dump(results_df, fout)
    fout.close()
    #save stats of run to file
    (prec,tp_norm,fp_norm)= precision(predict,TestY)
    (rec,tp_norm,fn_norm)= recall(predict,TestY)
    stats={}
    stats["prec"]= prec
    stats["rec"]= rec
    stats["fp_norm"]= fp_norm
    stats["fn_norm"]= fn_norm
    stats["frac_corr"]= fraction_correct(predict,TestY)
    fout = open(stats_save_name, 'w') 
    dump(stats, fout)
    fout.close()

fin=open('NoLims_shuffled_blob_features.pickle',"r")
feat_df= load(fin)
fin.close()
fin=open('NoLims_shuffled_url_answer.pickle',"r")
url_df= load(fin)
fin.close()

predict_save_name="NoLims_shuffled_blob_predict.pickle"
stats_save_name="NoLims_shuffled_blob_stats.pickle"
train_and_predict(feat_df,url_df,predict_save_name,stats_save_name)




