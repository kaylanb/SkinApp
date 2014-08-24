from instagram.client import InstagramAPI
import time
import matplotlib.pyplot as plt
import numpy as np
import urllib2, cStringIO
from PIL import Image


# time.sleep(1.5)
# api= InstagramAPI(client_id="1dcd8b9c15c24b7bb571ebe805b61c64",                  client_secret="f5c62b342e3540a3b7f84cd2fd725cfa")

# api.location_search(foursquare_v2_id="3fd66200f964a52018ed1ee3")
# mm, next = api.location_recent_media(location_id=797)
path="./training_image_urls/"
urlFile= "NewTraining_Food_everyones.txt"
urls = np.loadtxt(path+ urlFile, dtype="str")
# for url in urls:
url= urls[0]
read = urllib2.urlopen(url).read()
obj = Image.open(cStringIO.StringIO(read))
image = np.array(obj)

fig, axis = plt.subplots(2,2,figsize=(5,5))
ax=axis.flatten()
ax[0].imshow(image)
ax[1].imshow(image[...,0],cmap="gray")
ax[2].imshow(image[...,1],cmap="gray")
ax[3].imshow(image[...,2],cmap="gray")
titles=["rgb","r","g","b"]
for cnt,axi in enumerate(ax):
    axi.set_title(titles[cnt])
    [label.set_visible(False) for label in axi.get_xticklabels()]
    [label.set_visible(False) for label in axi.get_yticklabels()]
plt.show()

