#http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog/
import skinmap as sm
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import sqrt, pi, arctan2, cos, sin, ndimage, fftpack, stats
from skimage import exposure, measure, feature
import pandas as pd
from PIL import Image
import cStringIO
import urllib2
import numpy as np
from pylab import *

#http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog/

#labels_df = pd.DataFrame(columns=['blocks'])

#get url
file="../training_image_urls/NewTraining_Faces_everyones.txt" #'People_All.txt'
urls=np.loadtxt(file,dtype="str")
nrow = len(urls)
count = 0

#labels_df = np.zeros((nrow, 8100)) #People_Feature_7.csv Food_Feature_7.csv
#labels_df = np.zeros((nrow, 15390)) #People_Feature_8.csv Food_Feature_8.csv
#labels_df = np.zeros((nrow, 1296)) #People_Feature_9.csv Food_Feature_9.csv
#labels_df = np.zeros((nrow, 29241)) #People_Feature_10.csv Food_Feature_10.csv

#labels_df = np.zeros((nrow, 29241)) #People_All_1.csv Food_All_1.csv
#labels_df = np.zeros((nrow, 46656)) #People_All_2.csv Food_All_2.csv
#labels_df = np.zeros((nrow, 65025)) #People_All_3.csv Food_All_3.csv
#labels_df = np.zeros((nrow, 8100)) #People_All_4.csv Food_All_4.csv
#labels_df = np.zeros((nrow, 11664)) #People_All_5.csv Food_All_5.csv
#labels_df = np.zeros((nrow, 14400)) #People_All_6.csv Food_All_6.csv
#labels_df = np.zeros((nrow, 1296)) #People_All_7.csv Food_All_7.csv
#labels_df = np.zeros((nrow, 1296)) #People_All_8.csv Food_All_8.csv
labels_df = np.zeros((nrow, 900)) #People_All_9.csv Food_All_9.csv

for url in urls:
    print url,
    try:
        read= urllib2.urlopen(url).read()
    except urllib2.URLError:
        continue
    obj = Image.open( cStringIO.StringIO(read) )
    img = np.array(obj.convert('L'))
    #blocks = feature.hog(img, pixels_per_cell=(50, 50), cells_per_block=(3, 3), visualise=False, normalise=True) #People_Feature_7.csv Food_Feature_7.csv
    #blocks = feature.hog(img, orientations=9, pixels_per_cell=(50,30), cells_per_block=(3,3), visualise=False, normalise=True) #People_Feature_8.csv Food_Feature_8.csv
    #blocks = feature.hog(img, orientations=9, pixels_per_cell=(100,100), cells_per_block=(3,3), visualise=False, normalise=True) #People_Feature_9.csv Food_Feature_9.csv
    #blocks = feature.hog(img, orientations=9, pixels_per_cell=(30,30), cells_per_block=(3,3), visualise=False, normalise=True) #People_Feature_10.csv Food_Feature_10.csv

    #blocks = feature.hog(img, orientations=9, pixels_per_cell=(30,30), cells_per_block=(3,3), visualise=False, normalise=True) #People_All_1.csv Food_All_1.csv
    #blocks = feature.hog(img, orientations=9, pixels_per_cell=(30,30), cells_per_block=(4,4), visualise=False, normalise=True) #People_All_2.csv Food_All_2.csv
    #blocks = feature.hog(img, orientations=9, pixels_per_cell=(30,30), cells_per_block=(5,5), visualise=False, normalise=True) #People_All_3.csv Food_All_3.csv
    #blocks = feature.hog(img, orientations=9, pixels_per_cell=(50,50), cells_per_block=(3,3), visualise=False, normalise=True) #People_All_4.csv Food_All_4.csv
    #blocks = feature.hog(img, orientations=9, pixels_per_cell=(50,50), cells_per_block=(4,4), visualise=False, normalise=True) #People_All_5.csv Food_All_5.csv
    #blocks = feature.hog(img, orientations=9, pixels_per_cell=(50,50), cells_per_block=(5,5), visualise=False, normalise=True) #People_All_6.csv Food_All_6.csv
    #blocks = feature.hog(img, orientations=9, pixels_per_cell=(100,100), cells_per_block=(3,3), visualise=False, normalise=True) #People_All_7.csv Food_All_7.csv
    #blocks = feature.hog(img, orientations=9, pixels_per_cell=(100,100), cells_per_block=(4,4), visualise=False, normalise=True) #People_All_8.csv Food_All_8.csv
    blocks = feature.hog(img, orientations=9, pixels_per_cell=(100,100), cells_per_block=(5,5), visualise=False, normalise=True) #People_All_9.csv Food_All_9.csv
    print "hogs done"

    #labels_df.loc[len(labels_df.index)]=([blocks])
    #if(len(blocks) == 29241): #People_Feature_10.csv Food_Feature_10.csv
    #if(len(blocks) == 1296): #People_Feature_9.csv Food_Feature_9.csv
    #if(len(blocks) == 15390): #People_Feature_8.csv Food_Feature_8.csv
    #if(len(blocks) == 8100): #People_Feature_7.csv Food_Feature_7.csv

    #if(len(blocks) == 29241): #People_All_1.csv Food_All_1.csv
    #if(len(blocks) == 46656): #People_All_2.csv Food_All_2.csv
    #if(len(blocks) == 65025): #People_All_3.csv Food_All_3.csv
    #if(len(blocks) == 8100): #People_All_4.csv Food_All_4.csv
    #if(len(blocks) == 11664): #People_All_5.csv Food_All_5.csv
    #if(len(blocks) == 14400): #People_All_6.csv Food_All_6.csv
    #if(len(blocks) == 1296): #People_All_7.csv Food_All_7.csv
    #if(len(blocks) == 1296): #People_All_8.csv Food_All_8.csv
    if(len(blocks) == 900): #People_All_9.csv Food_All_9.csv
        labels_df[count] = blocks
    count += 1
    
#labels_df.to_csv("Food_Features_10.csv")
np.savetxt("People_All_9.csv", labels_df, delimiter=",")
