import pandas as pd
from PIL import Image
import cStringIO
import urllib2
import numpy as np

#get url
file="../training_image_urls/NewTraining_Faces_everyones.txt" #'People_All.txt'
urls=np.loadtxt(file,dtype="str")
url_good_list=[]

for url in urls:
    try:
        read= urllib2.urlopen(url).read()
        url_good_list.append(url)
    except urllib2.URLError:
        continue

urls_df= pd.DataFrame()
urls_df["urls"]= url_good_list
np.savetxt("Faces_urls.csv", labels_df, delimiter=",")
