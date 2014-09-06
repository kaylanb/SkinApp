'''outputs features for HOG machine learning'''
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import sqrt, pi, arctan2, cos, sin, ndimage, fftpack, stats
from skimage import exposure, measure, feature
import pandas as pd
from PIL import Image
import cStringIO
import urllib2
import numpy as np

def OutputFeatures_ForOneImage(rgb_image,image_name):
    feat = np.zeros(900) #People_All_9.csv Food_All_9.csv
    blocks = feature.hog(rgb_image, orientations=9, pixels_per_cell=(100,100), cells_per_block=(5,5), visualise=False, normalise=True) #People_All_9.csv Food_All_9.csv

    if(len(blocks) == 900): #People_All_9.csv Food_All_9.csv
        feat = blocks

    name_df=pd.DataFrame()
    name_df["image_name"]= image_name
    feat_df= pd.DataFrame(feat)
    final_df=pd.concat([name_df,feat_df],axis=1) 
    final_df.to_csv("tmp/HogFeatures.csv")

def Hog_predict_UploadImage(rgb_image,image_name):
    OutputFeatures_ForOneImage(rgb_image,image_name)
    feat_df= pd.read_csv("tmp/HogFeatures.csv")
    feat= feat_df.ix[].values

    root= "machine_learn/HOG/csv_features/"
    Food_df = read_csv(root+"hog_features_9_NewTraining_Food_everyones.csv')
    People_df = read_csv(root+"hog_features_9_NewTraining_Faces_everyones.csv')
    
    cTrainF = rand(len(Food_df)) > .5
    cTestF = ~cTrainF
    cTrainP = rand(len(People_df)) > .5
    cTestP = ~cTrainP
    
    TrainX_df = pd_concat([People_df[cTrainP], Food_df[cTrainF]],axis=0)
    TrainX= TrainX_df.ix[:,2:].values
    TrainY = concatenate([ones(len(People_df[cTrainP])), zeros(len(Food_df[cTrainF]))])

    ET_classifier = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0)
    ET_classifier.fit(TrainX,TrainY)
    ET_prediction = ET_classifier.predict(feat) 

    LinSVC_classifier = svm.LinearSVC()
    LinSVC_classifier.fit(TrainX,TrainY)
    LinSVC_prediction = LinSVC_classifier.predict(feat)

    print "predict from ET: %d, SVC: %d" % (ET_prediction,LinSVC_prediction)

#testing
# import matplotlib.image as mpimg
# file='uploads/engage_1ps.jpg'
# img= mpimg.imread(file)
file="machine_learn/training_image_urls/NewTraining_Food_everyones.txt"
urls=np.loadtxt(file,dtype="str")
url=urls[10]
read= urllib2.urlopen(url).read()
obj = Image.open( cStringIO.StringIO(read) )
img = np.array(obj.convert('L'))
import pylab as py
# py.imshow(img)
# py.show()
# OutputFeatures_ForOneImage(img,file)
Hog_predict_UploadImage(img,file)