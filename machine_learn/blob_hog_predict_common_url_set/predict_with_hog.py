'''outputs features for HOG machine learning'''
from matplotlib import image as mpimg
from scipy import sqrt, pi, arctan2, cos, sin, ndimage, fftpack, stats
from skimage import exposure, measure, feature
from PIL import Image
import cStringIO
import urllib2

from numpy.random import rand
from numpy import ones, zeros, concatenate, array
from pandas import read_csv, DataFrame
from pandas import concat as pd_concat
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
import pickle

def get_indices_urls_that_exist(urls): 
    inds_exist=[]
    cnt=-1
    for url in df.URL.values:
        cnt+=1
        print "cnt= %d" % cnt
        try:
            read= urllib2.urlopen(url).read()
            inds_exist.append(cnt)
        except urllib2.URLError:
            continue
    return np.array(inds_exist)

def update_dataframes_with_urls_exist():
    root= "machine_learn/blob_hog_predict_common_url_set/"
    url_1= root+"NoLims_shuffled_url_answer.pickle"
    fin=open(url_1,"r")
    url_df= pickle.load(fin)
    fin.close()
    url_1_inds= get_indices_urls_that_exist(url_df.URL.values)

    path=root+"NoLims_shuffled_blob_features.pickle"
    fin=open(path,"r")
    blob_feat_df= pickle.load(fin)
    fin.close()

    url_df_exist= url_df.ix[url_1_inds,:]
    blob_feat_df_exist= blob_feat_df.ix[url_1_inds,:]

    fout = open("NoLims_shuffled_url_answer.pickle", 'w') 
    pickle.dump(url_df_exist, fout)
    fout.close()
    fout = open("NoLims_shuffled_blob_features.pickle", 'w') 
    pickle.dump(blob_feat_df_exist, fout)
    fout.close()

def hog_features(urls,output_csv_name):
    feat = zeros((len(urls), 900))
    count=0
    for url in urls:
        print "count= %d" % count
        read= urllib2.urlopen(url).read()
        obj = Image.open( cStringIO.StringIO(read) )
        img = array(obj.convert('L'))
    
        blocks = feature.hog(img, orientations=9, pixels_per_cell=(100,100), cells_per_block=(5,5), visualise=False, normalise=True) #People_All_9.csv Food_All_9.csv
        if(len(blocks) == 900):
            feat[count] = blocks
        count += 1

    feat_df= DataFrame(feat)
    final_df.to_csv(output_csv_name)
    return 

update_lists_with_urls_exist()
# fin=open('FacesAndLimbs_shuffled_url_answer.pickle',"r")
# df= pickle.load(fin)
# fin.close()
# urls=df.URL.values
# hog_features(urls, "FacesAndLimbs_Food_hog_features.csv")

# def predict_TestData(Food_df,People_df):
#     cTrainF = rand(len(Food_df)) > .5
#     cTestF = ~cTrainF
#     cTrainP = rand(len(People_df)) > .5
#     cTestP = ~cTrainP
# 
#     TrainX_df = pd_concat([People_df[cTrainP], Food_df[cTrainF]],axis=0)
#     TestX_df = pd_concat([People_df[cTestP], Food_df[cTestF]],axis=0)
# 
#     TrainX= TrainX_df.ix[:,2:].values
#     TestX= TestX_df.ix[:,2:].values
#     TrainY = concatenate([ones(len(People_df[cTrainP])), zeros(len(Food_df[cTrainF]))])
#     TestY = concatenate([ones(len(People_df[cTestP])), zeros(len(Food_df[cTestF]))])
# 
#     ET_classifier = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0)
#     ET_classifier.fit(TrainX,TrainY)
#     ET_prediction = ET_classifier.predict(TestX) 
# 
#     LinSVC_classifier = svm.LinearSVC()
#     LinSVC_classifier.fit(TrainX,TrainY)
#     LinSVC_predict = LinSVC_classifier.predict(TestX)
# 
#     a=DataFrame()
#     a["url"]=TestX_df.urls.values
#     a["answer"]=TestY
#     a["ET_predict"]=ET_prediction
#     a["LinSVC_predict"]=LinSVC_predict
#     a.to_csv("prediction_for_TestData.csv")
# 
# Food_df = read_csv('FacesAndLimbs_Food_hog_features.csv')
# People_df = read_csv('FacesAndLimbs_People_hog_features.csv')
# predict_TestData(Food_df,People_df)


