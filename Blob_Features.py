'''Use Todd's skinmap: SkinLikelihood() and Kaylan's tested equivalent functions for skimage measure.regionprops() to generate ML features (ie it is not clear measure.regionprops() does what it says it does for each label!?)'''

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy import ndimage, fftpack, stats
from skimage import exposure, measure,feature,filter
from pandas import DataFrame, read_csv
import numpy as np
from skimage.filter import sobel
from skimage.morphology import convex_hull_image
from scipy.spatial import ConvexHull
import matplotlib.patches
import pickle
#personal modules
import skinmap as sm

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
    return prec,tp_norm,fp_norm

def recall(predict,ans):
    rec= tp(predict,ans)/(tp(predict,ans) + fn(predict,ans))
    tp_norm= tp(predict,ans)/totp(ans)
    fn_norm= fn(predict,ans)/totn(ans)
    return rec,tp_norm,fn_norm

def fraction_correct(predict,ans):
    tp= float( len(np.where(predict.astype('bool') & ans.astype('bool'))[0]) )
    tn= float( len(np.where( (predict.astype('bool')==False) & (ans.astype('bool')==False) )[0]) )
    return (tp+tn)/len(ans)

def SepPts((x1,y1),(x2,y2)):
        return np.sqrt( (x2-x1)**2 + (y2-y1)**2 )

def percent_area(blob_area,image_size):
    return float(blob_area)/image_size

def percent_vert_horiz_lines(FilledBlobImg,props_area):
    v = filter.vsobel(FilledBlobImg)
    v_floor = filter.threshold_otsu(v)
    ind_v= np.where(v>v_floor)
    h= filter.hsobel(FilledBlobImg)
    h_floor = filter.threshold_otsu(h)
    ind_h= np.where(h>h_floor)
    
    vert_and_horiz=np.zeros(v.shape).astype('bool')
    vert_and_horiz[ind_v]= True
    vert_and_horiz[ind_h]= True
    ind= np.where(vert_and_horiz)[0]
    return float(ind.size)/props_area
    
def skin_pix_percent_filled(blob,filled_blob):
    diff= filled_blob.astype('int')-blob.astype('int')
    return float( len(np.where(diff ==1)[0])) /len(np.where(blob.astype('bool'))[0])

def hull_percent_filled(filled_blob):
    edge = sobel(filled_blob)
    hull_img = convex_hull_image(edge)
    return float( len(np.where(filled_blob.astype('bool'))[0])) /len(np.where(hull_img.astype('bool'))[0])

def residualSum_EllipseFit_to_HullEdge(edge_image_of_filled_blob,props_x_centroid,props_y_centroid):
    ind=np.where(edge_image_of_filled_blob.astype("bool"))
    x_pts=ind[1]
    y_pts=ind[0]
    pts=np.zeros((len(x_pts),2))-1
    pts[:,0]= x_pts
    pts[:,1]= y_pts
    
    hull = ConvexHull(pts)
    hull_pts=[]
    for simplex in hull.simplices:
        hull_pts.append([pts[simplex,0][0],pts[simplex,1][0]] )  #pts[simplex,0] is [x1,x2], want [x1,y1]

    hull_pts=np.array(hull_pts)
    elMod= measure.EllipseModel()
    elMod.estimate(hull_pts)
    return elMod.residuals(hull_pts).sum()

class GetFloorsObj():
    def __init__(self,image):
        self.HardAreaCutoff= 100. 
        self.AreaFloor= 0.005*image.size
        self.SkinPercentGotFilledFloor= 0.15
        self.VertHorizCeil = 0.3
        self.HullPercentFilledFloor= 0.5
        
def getBlobsPropsIndex(areaImg,props,props_indices):
    im= areaImg.copy()
    ind_all= set(range(len(props)))
    ind_keep= set(props_indices)
    ind_rm= ind_all.difference(ind_keep)
    for cnt in ind_rm:  
        ind_blob= np.where(im == props[cnt].label)
        im[ind_blob]=0.
    return im 

def percent_area_blobs_in_list(props,indices,image):
    area=0.
    for cnt in indices:
        area += props[cnt].area
    return float(area)/image.size

class BlobFeaturesToPlot():
    pass

def PlotBlobFeatures(Figs):
    '''multiplot showing blobs remaining once features isolate blobs
    **two features extracted for each "feature image", Number of blobs and Percent area those blobs occupy of image '''
    f,axis=plt.subplots(4,2,figsize=(5,10))
    ax=axis.flatten()
    ax[0].imshow(Figs.rgb_image)
    ax[1].imshow(Figs.SkinLikelihood_6,cmap="gray")
    ax[2].imshow(Figs.areaImg_AllBlobs,cmap="gist_ncar")
    ax[3].imshow(Figs.areaImg_AreaGrHardCutoff,cmap="gist_ncar")
    ax[4].imshow(Figs.AreaGrFloor, cmap="gray")
    ax[5].imshow(Figs.SkinPixFilledGrFloor, cmap="gray")
    ax[6].imshow(Figs.AmtHorizVertLessCeil, cmap="gray")
    ax[7].imshow(Figs.HullFilledGreaterFloor, cmap="gray")
    f.subplots_adjust(wspace=0.5,hspace=0.2)
    titles=["image","SkinPixels","Blobs","Blobs_GrMinArea","Blobs_GrUserFloor",\
            "Blobs_wSkinFilledGrFloor","Blobs_wSmallHorizVertLines",\
            "Blobs_wConvexHullAreaGrFloor"]
    for cnt,title in enumerate(titles):
        ax[cnt].set_title(title,fontdict={'fontsize':10.})
        [label.set_visible(False) for label in ax[cnt].get_xticklabels()]
        [label.set_visible(False) for label in ax[cnt].get_yticklabels()]
    plt.savefig("tmp/BlobFeaturesPlot.png",bbox_inches='tight')
#     plt.show()
    

def extract_features_and_feature_Figs_to_plot(n_images,nth_image,rgb_image,image_name):
    cols= [
    'UserUploadedImageName',
    'percent_skin_SkinLikelihood_6',
    'percent_skin_abg',
    'percent_skin_cbcr',
    'percent_skin_cccm',
    'percent_skin_equalized_abg',
    'percent_skin_equalized_cbcr',
    'percent_skin_equalized_cccm',
    'percent_skin_adapt_abg',
    'percent_skin_adapt_cbcr',
    'percent_skin_adapt_cccm',
    'N_area_greater_HardCutoff',
    'N_area_greater_floor',
    'N_skin_pix_filled_greater_floor',
    'N_amt_horiz_vert_lines_less_ceil',
    'N_hull_filled_greater_floor',
    'percent_area_gr_HardCutoff',
    'percent_area_gr_floor',
    'percent_area_skinpix_gr_floor',
    'percent_area_horiz_vert_less_ceil',
    'percent_area_hull_filled_gr_floor'
    ]

    labels_df=DataFrame()
    for name in cols:
        labels_df[name] = np.zeros(n_images)
    labels_df["UserUploadedImageName"]= image_name

    Figs= BlobFeaturesToPlot() #store blob feature images for imshow
    Figs.rgb_image= rgb_image

    obj= sm.SetUpImage(rgb_image)
    im= obj.SkinLikelihood()
    SkinFilter= FilterSlice(im,6)  #6 is best value, 7 too retrictive, < 6 not great
    Figs.SkinLikelihood_6 = SkinFilter
    
    #percent of skin-like pixels in each skinmap
    labels_df['percent_skin_SkinLikelihood_6'][nth_image] = sm.percent_skin(SkinFilter)
    labels_df['percent_skin_abg'][nth_image] = sm.percent_skin(obj.ImgReg.skin.abg)
    labels_df['percent_skin_cbcr'][nth_image] = sm.percent_skin(obj.ImgReg.skin.cbcr)
    labels_df['percent_skin_equalized_cccm'][nth_image] = sm.percent_skin(obj.ImgReg.skin.cccm)
    labels_df['percent_skin_equalized_abg'][nth_image] = sm.percent_skin(obj.ImgEq.skin.abg)
    labels_df['percent_skin_equalized_cbcr'][nth_image] = sm.percent_skin(obj.ImgEq.skin.cbcr)
    labels_df['percent_skin_equalized_cccm'][nth_image] = sm.percent_skin(obj.ImgEq.skin.cccm)
    labels_df['percent_skin_adapt_abg'][nth_image] = sm.percent_skin(obj.ImgEqAdapt.skin.abg)
    labels_df['percent_skin_adapt_cbcr'][nth_image] = sm.percent_skin(obj.ImgEqAdapt.skin.cbcr)
    labels_df['percent_skin_adapt_cccm'][nth_image] = sm.percent_skin(obj.ImgEqAdapt.skin.cccm)
    
    image= SkinFilter
    fl= GetFloorsObj(image)
    #find blobs, consider only those containing > 900 skin pixels
    labels,n =ndimage.measurements.label(image,np.ones((3,3)))
    area = ndimage.measurements.sum(image,labels, index=np.arange(labels.max() + 1))
    areaImg = area[labels]
    Figs.areaImg_AllBlobs= areaImg
    #cutoff, blobs must be sufficiently large
    ind_cutoff= np.where(areaImg < fl.HardAreaCutoff)
    areaImg[ind_cutoff] = 0.
    Figs.areaImg_AreaGrHardCutoff= areaImg
    props= measure.regionprops(areaImg.astype('int'),cache=True)
    
    ind_area_greater_floor=[]
    ind_skin_pix_filled_greater_floor=[]
    ind_amt_horiz_vert_lines_less_ceil=[]
    ind_hull_filled_greater_floor=[]
    for cnt in range(len(props)):
        blob= np.zeros(areaImg.shape).astype('bool')
        ind_blob= np.where(areaImg == props[cnt].label)
        blob[ind_blob]=True       
        filled_blob= ndimage.morphology.binary_fill_holes(blob)
        
        if props[cnt].area > fl.AreaFloor: 
            ind_area_greater_floor.append(cnt)
        if skin_pix_percent_filled(blob,filled_blob) > fl.SkinPercentGotFilledFloor:
            ind_skin_pix_filled_greater_floor.append(cnt)
        if percent_vert_horiz_lines(filled_blob,props[cnt].area) < fl.VertHorizCeil: 
            ind_amt_horiz_vert_lines_less_ceil.append(cnt)
        if hull_percent_filled(filled_blob) > fl.HullPercentFilledFloor:
            ind_hull_filled_greater_floor.append(cnt)
            
    Figs.AreaGrFloor= getBlobsPropsIndex(areaImg,props,ind_area_greater_floor)
    Figs.SkinPixFilledGrFloor= getBlobsPropsIndex(areaImg,props,ind_skin_pix_filled_greater_floor)
    Figs.AmtHorizVertLessCeil= getBlobsPropsIndex(areaImg,props,ind_amt_horiz_vert_lines_less_ceil)
    Figs.HullFilledGreaterFloor= getBlobsPropsIndex(areaImg,props,ind_hull_filled_greater_floor)
    
    #fill labels
    labels_df['N_area_greater_HardCutoff'][nth_image]= len(props)
    labels_df['N_area_greater_floor'][nth_image]= len(ind_area_greater_floor)
    labels_df['N_skin_pix_filled_greater_floor'][nth_image]= len(ind_skin_pix_filled_greater_floor)
    labels_df['N_amt_horiz_vert_lines_less_ceil'][nth_image]= len(ind_amt_horiz_vert_lines_less_ceil)
    labels_df['N_hull_filled_greater_floor'][nth_image]= len(ind_hull_filled_greater_floor)
    #labels_df['EllipseFitResid_4HullFilled'][nth_image]= EllipseFitResid_4HullFilled
    #total percent area of all blobs in each grouping
    labels_df['percent_area_gr_HardCutoff'][nth_image]= percent_area_blobs_in_list(props,range(len(props)),SkinFilter)
    labels_df['percent_area_gr_floor'][nth_image]= percent_area_blobs_in_list(props,ind_area_greater_floor,SkinFilter)
    labels_df['percent_area_skinpix_gr_floor'][nth_image]= percent_area_blobs_in_list(props,ind_skin_pix_filled_greater_floor,SkinFilter)
    labels_df['percent_area_horiz_vert_less_ceil'][nth_image]= percent_area_blobs_in_list(props,ind_amt_horiz_vert_lines_less_ceil,SkinFilter)
    labels_df['percent_area_hull_filled_gr_floor'][nth_image]= percent_area_blobs_in_list(props,ind_hull_filled_greater_floor,SkinFilter)
    
    return labels_df,Figs
    

def FilterSlice(SkinFilter,IntVal):
    im=np.zeros(SkinFilter.shape)
    ind= np.where(SkinFilter.astype('int') ==IntVal)
    im[ind]=1
    return im

def ComputeBlobFeatures(rgb_image,image_name):
    if rgb_image.shape[2] != 3:
        print "IMAGE IS NOT RGB, crash..."
        raise ValueError
    n_images=1
    nth_image=0
    features_df,Figs= extract_features_and_feature_Figs_to_plot(\
                                        n_images,nth_image,rgb_image,image_name)
    features_df.to_csv("tmp/BlobFeatures.csv")
    PlotBlobFeatures(Figs)

class MLResults():
    def __init__(self):
        self.HasPeople= -1
        self.frac_correct=-1
        self.precision=-1
        self.recall=-1
        self.tp_norm=-1
        self.fp_norm=-1

def BlobMethod_on_Image(rgb_image,image_name):
    #create feature file and feature explanatory image plot: 
    #'tmp/BlobFeatures.csv' and 'tmp/BlobFeaturesPlot.png'
    ComputeBlobFeatures(rgb_image,image_name)
    #predict based on features
    features_df = read_csv('tmp/BlobFeatures.csv')
    features_need=features_df.ix[0,2:]
    image_features= features_need.values
    #load trained Random Forest Classifier
    fin=open("machine_learn/Blob/RandForestTrained.pickle","r")
    RF=pickle.load(fin)
    fin.close()
    image_predict = RF.predict(image_features)
    #store results
    blob_results= MLResults()
    if image_predict.astype('int')[0] == 0: blob_results.HasPeople= True
    elif image_predict.astype('int')[0] == 1: blob_results.HasPeople= False
    else: raise ValueError
    #get accuracy of Blob Method
    f_results= "machine_learn/Blob/TestImageSet_predictions_answers.pickle"
    try: 
        fin=open(f_results,"r")
        results_df=pickle.load(fin)
        fin.close()
    except IOError:
        print "ERROR: %s does not exist, cannot determine accuracy of Blob Method" % f_results
        print "try running: 'python machine_learn/Build_ML_Results/build.py' "
    predict= results_df.predict.values
    answer= results_df.answer.values
    (prec,tp_norm,fp_norm)= precision(predict,answer)
    (rec,tp_norm,fn_norm)=  recall(predict,answer)
    frac_correct= fraction_correct(predict,answer)
    #store results
    blob_results.frac_correct= frac_correct
    blob_results.precision=prec
    blob_results.recall= rec
    blob_results.tp_norm= tp_norm
    blob_results.fp_norm= fp_norm
    return blob_results


###testing 
# import matplotlib.image as mpimg
# file='uploads/engage_1ps.jpg'
# img= mpimg.imread(file)
# blob_results= BlobMethod_on_Image(img,file)
# print "blob_results.HasPeople= %s" % blob_results.HasPeople


