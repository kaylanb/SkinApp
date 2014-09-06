import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from scipy import sqrt, pi, arctan2, cos, sin, ndimage, fftpack, stats
from skimage import exposure, measure, feature
from pandas import read_csv, DataFrame
from pandas import concat as pd_concat
from PIL import Image
import cStringIO
import urllib2
from numpy.random import rand
from numpy import ones, zeros, concatenate

from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm

def save_ImageOfHog(grey_img,ImageOfHog,rgb_img):
    f,ax=plt.subplots(1,3,figsize=(20,20))
    ax[0].imshow(rgb_img)
    ax[1].imshow(grey_img,cmap='gray')
    ax[2].imshow(ImageOfHog,cmap='gray',interpolation='nearest')
    titles=["RGB Image","Greyscale input for HOG","Image of Histogram Oriented Gradients (HOG)"]
    for cnt,axis in enumerate(ax):
        axis.set_title(titles[cnt],fontdict={'fontsize':20.})
        [label.set_visible(False) for label in axis.get_xticklabels()]
        [label.set_visible(False) for label in axis.get_yticklabels()]
    plt.savefig("tmp/image_of_hog.png",dpi=200)

def CalcHog_FeaturesAndImage_ForOneImage(grey_img,image_name,rgb_img):
    feat = zeros((1,900)) #People_All_9.csv Food_All_9.csv
    #get hog features
    blocks = feature.hog(grey_img, orientations=9, pixels_per_cell=(100,100), cells_per_block=(5,5), visualise=False, normalise=True) #People_All_9.csv Food_All_9.csv
    #slightly diff params for better hog visualization
    junk_block,ImageOfHog=feature.hog(grey_img, pixels_per_cell=(10,10), cells_per_block=(30,30),visualise=True,normalise=True)
    
    if(len(blocks) == 900): #People_All_9.csv Food_All_9.csv
        feat[0] = blocks

    name_df=DataFrame()
    name_df["image_name"]= image_name
    feat_df= DataFrame(feat)
    final_df=pd_concat([name_df,feat_df],axis=1) 
    final_df.to_csv("tmp/HogFeatures.csv")
    
    save_ImageOfHog(grey_img,ImageOfHog,rgb_img)

def Hog_predict_UploadImage(grey_img,image_name,rgb_img):
    CalcHog_FeaturesAndImage_ForOneImage(grey_img,image_name,rgb_img)
    feat_df= read_csv("tmp/HogFeatures.csv")
    feat_vals= feat_df.ix[:,2:].values

    root= "machine_learn/HOG/csv_features/"
    Food_df = read_csv(root+"hog_features_9_NewTraining_Food_everyones.csv")
    People_df = read_csv(root+"hog_features_9_NewTraining_Faces_everyones.csv")
    
    cTrainF = rand(len(Food_df)) > .5
    cTestF = ~cTrainF
    cTrainP = rand(len(People_df)) > .5
    cTestP = ~cTrainP
    
    TrainX_df = pd_concat([People_df[cTrainP], Food_df[cTrainF]],axis=0)
    TrainX= TrainX_df.ix[:,2:].values
    TrainY = concatenate([ones(len(People_df[cTrainP])), zeros(len(Food_df[cTrainF]))])

    ET_classifier = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0)
    ET_classifier.fit(TrainX,TrainY)
    ET_prediction = ET_classifier.predict(feat_vals) 

    LinSVC_classifier = svm.LinearSVC()
    LinSVC_classifier.fit(TrainX,TrainY)
    LinSVC_prediction = LinSVC_classifier.predict(feat_vals)
    return ET_prediction, LinSVC_prediction

### testing
# file="machine_learn/training_image_urls/NewTraining_Food_everyones.txt"
# urls=np.loadtxt(file,dtype="str")
# url=urls[11]
# read= urllib2.urlopen(url).read()
# obj = Image.open( cStringIO.StringIO(read) )
# rgb_img= np.array(obj)
# grey_img = np.array(obj.convert('L'))
# Hog_predict_UploadImage(grey_img,file,rgb_img)