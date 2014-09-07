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

def hog_features(ans_url_df,output_pickle_name):
    urls=ans_url_df.URL.values
    answers=ans_url_df.answer.values
    
    urls_exist=[]
    ans_exist=[]
    cnt=-1
    for url,ans in zip(urls,answers):
        cnt+=1
        print "cnt= %d , checking urls" % cnt
        try:
            read= urllib2.urlopen(url).read()
            urls_exist.append(url)
            ans_exist.append(ans)
        except urllib2.URLError:
            continue

    urls_exist= array(urls_exist)
    ans_exist= array(ans_exist)
    feat = zeros((len(urls_exist), 900))
    count=0
    for url in urls_exist:
        print "count= %d -- calc features" % count
        read= urllib2.urlopen(url).read()
        obj = Image.open( cStringIO.StringIO(read) )
        img = array(obj.convert('L'))
    
        blocks = feature.hog(img, orientations=9, pixels_per_cell=(100,100), cells_per_block=(5,5), visualise=False, normalise=True) #People_All_9.csv Food_All_9.csv
        if(len(blocks) == 900):
            feat[count] = blocks
        count += 1

    urls_exist_df= DataFrame(urls_exist,columns=["URL"])
    ans_exist_df= DataFrame(ans_exist,columns=["answer"])
    feat_df= DataFrame(feat)
    final_df= pd_concat([urls_exist_df,ans_exist_df,feat_df],axis=1)
    fout = open(output_pickle_name, 'w') 
    pickle.dump(final_df.dropna(), fout)
    fout.close()

def train_and_predict(feat_df,predict_save_name):
    TrainX= feat_df.values[0:300,2:]
    TrainY= feat_df.answer.values[0:300]
    TestX= feat_df.values[300:,2:]
    TestY=feat_df.answer.values[300:]
    TestUrls= feat_df.URL.values[300:]

    ET_classifier = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0)
    ET_classifier.fit(TrainX,TrainY)
    ET_prediction = ET_classifier.predict(TestX) 

    LinSVC_classifier = svm.LinearSVC()
    LinSVC_classifier.fit(TrainX,TrainY)
    LinSVC_predict = LinSVC_classifier.predict(TestX)

    a=DataFrame()
    a["url"]=TestUrls
    a["answer"]=TestY
    a["ET_predict"]=ET_prediction
    a["LinSVC_predict"]=LinSVC_predict
    fout = open(predict_save_name, 'w') 
    pickle.dump(a, fout)
    fout.close()

###hog features
# fin=open('FacesAndLimbs_shuffled_url_answer.pickle',"r")
# ans_url_df= pickle.load(fin)
# fin.close()
# hog_features(ans_url_df, "FacesAndLimbs_shuffled_hog_features.pickle")

###hog predict
fin=open('NoLims_shuffled_hog_features.pickle',"r")
feat_df= pickle.load(fin)
fin.close()
train_and_predict(feat_df,"NoLims_shuffled_hog_predict.pickle")


