from flask import Flask
from flask import render_template, flash, redirect
import numpy as np
#forms
from flask.ext.wtf import Form
from wtforms import TextField, SubmitField, TextAreaField, BooleanField, SelectField,SelectMultipleField
from wtforms.validators import Required, Optional
#other
from pandas import read_csv, DataFrame

#file upload modules
import os
from flask import request, url_for, send_from_directory
from werkzeug import secure_filename

#for ML results on images
import matplotlib.image as mpimg
from PIL import Image
import sys
import pickle
from random import randrange


app = Flask(__name__)
app.config.from_object('config')

# class Click2Play(Form):
#     type=TextField('Anything', validators = [Required()])
#     play= SubmitField("Click2Play")
# 
# class getWhatUserSees(Form):
#     UserSees = TextField('UserSees', validators = [Required()])
#     test= TextAreaField("test",validators= [Optional()])
#     MlOnImage= SubmitField("MachineLearnOnImage",validators = [Required()])
#    
class SelectOption(Form):
    choices= [("1","Use Test Image"),('2','Upload my own image')]
    HowAnalyze= SelectField("HowAnalyze",choices=choices)
    choices= [('1','Blob'),('2','HOG'),('3','Blob features'),('4','HOG features')]
    WhichMethods= SelectMultipleField("Which Methods",choices=choices)
# 
# class UploadAnalyze(Form):
#     choices= [('1','Blob'),('2','HOG'),('3','Blob features'),('4','HOG features')]
#     HowAnalyze= SelectField("HowAnalyze",choices=choices)

# def do_ML():
#     import  machine_learn as ml
#     (ans,predict,url) = ml.run()
#     try: 
#         f=open("tmp/ml_results.csv","r")
#         flash("results.csv already exists")
#     except IOError:
#         #if get here, file does not exist
#         df= DataFrame()
#         df["ans"]=ans
#         df["predict"]=predict
#         df["url"]=url
#         df.to_csv("tmp/ml_results.csv",index=True)
#         flash("output results.csv")

@app.route('/', methods=['GET','POST'])
# @app.route('/index', methods = ['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/AboutUs')
def AboutUs():
    return render_template('AboutUs.html')

# @app.route('/AnalyzeImg', methods = ['GET', 'POST'])
# def AnalyzeImg():
#     form= getWhatUserSees()
#     path="training_image_urls/"
#     urlFile= "NewTraining_Food_everyones.txt"
#     urls = np.loadtxt(path+ urlFile, dtype="str")
#     if form.validate_on_submit():
#         flash("HERE I AM")
#         flash('User Sees: %s, test= %s' % \
#             (form.UserSees.data))
#         return redirect('/ML')#,img_url=urls[0])
#     return render_template('AnalyzeImg.html',
#         title = 'AnalyzeImg',urls=urls,form=form)

#save stuff to tmp/ directory as .pickle file so can be loaded as necessary later
def save_to_tmp(obj,save_name):
    path='tmp/'+save_name
    fout = open(path, 'w') 
    pickle.dump(obj, fout)
    fout.close()

#load stuff from tmp/ directory that saved there previously
def load_from_tmp(f_name):
    path='tmp/'+f_name  
    fin=open(path,"r")
    obj=pickle.load(fin)
    fin.close()
    return obj

@app.route('/ShowTestData', methods = ['GET', 'POST'])
def ShowTestData():
    #get urls of test images, and urls that hog and blob predict have in common
    root="machine_learn/blob_hog_predict_common_url_set/"
    f_hog=root+"NoLims_shuffled_hog_predict.pickle"
    f_blob=root+"NoLims_shuffled_blob_predict.pickle"
    fin=open(f_hog,"r")
    hog_df= pickle.load(fin)
    fin.close()
    fin=open(f_blob,"r")
    blob_df= pickle.load(fin)
    fin.close()
    a=set(blob_df.url.values.flatten())
    b=set(hog_df.url.values.flatten())
    c=a.intersection(b)
    common_urls=np.array(list(c))
    if request.method == 'POST':  
        #if here then user clicked output blob hog for image
        if request.method == 'POST' and request.form["submit"]=="yes":  
            #if here then user clicked do again for new image
            return redirect('/ShowTestData') #url_for('upload_analyze'))
        url=load_from_tmp("url.pickle") #have to save+load this b/c random index
        #hog, blob prediction for url
        ans_pred_d= {} 
        index= np.where(blob_df.url == url)[0][0]
        ans_pred_d["ans"]= blob_df.answer.values[index]
        ans_pred_d["blob_pred"]=blob_df.predict.values[index]
        index= np.where(hog_df.url == url)[0][0]
        ans_pred_d["hog_pred_et"]= hog_df.ET_predict.values[index]
        ans_pred_d["hog_pred_svc"]= hog_df.LinSVC_predict.values[index]
        #blob stats
        fin=open(root+'NoLims_shuffled_blob_stats.pickle')
        blob_stats=pickle.load(fin)
        fin.close()
        #hog stats from 10 Trials through Train/Test set
        fin=open('machine_learn/HOG/hog_stats_10.pickle',"r")
        hog_stats=pickle.load(fin)
        fin.close()
        return render_template('show_user_test_data_blob_hog.html',\
                    url=url,WhatSee=request.form["WhatSee"],ans_pred_d=ans_pred_d,\
                    blob_stats=blob_stats,hog_stats=hog_stats)
    #AFTER if request.method so don't get new rand_index
    rand_index= randrange(len(common_urls))
    print "Above url set!? rand_index= %d" % rand_index
    url=common_urls[ rand_index ]
    save_to_tmp(url,"url.pickle")#save to tmp so can load it from inside ''if request.method=='POST' ''
    return render_template('show_user_test_data.html',\
                    url=url)



# @app.route('/OutputResults', methods = ['GET', 'POST'])
# def OutputResults():
#     do_ML()


# @app.route('/ML', methods = ['GET', 'POST'])#img_url=image_url)
# def ML():
#     f=open("tmp/blob.txt","r")
#     blob_txt= f.read()
#     f.close()
#     f=open("tmp/hog.txt","r")
#     hog_txt= f.read()
#     f.close()
# #     urls=request.args.get('urls')
# #     flash(urls[0] )
#     #fig out how pass: UserSaw=form.UserSees.data and image_url into this function!\
#     return render_template('ML.html',title = 'ML Results', \
#             blob_image_results="tmp/blob.png",blob_text_results=blob_txt, \
#             hog_image_results="tmp/hog.png",hog_text_results=hog_txt)

#################### upload file code
# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

#shows html form for user to upload local file
@app.route('/upload')
def upload():
    return render_template('upload.html')

#follows upload(), receives uploaded file
@app.route('/upload_process', methods=['POST'])
def upload_process():
    # Get the name of the uploaded file
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        saved_at=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(saved_at)
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
#         args={}
#         args["filename"]=filename
#         args["saved_at"]=saved_at
        return redirect( url_for('upload_analyze',filename=filename))#,saved_at) )

#if call "url_for('get_image_url',arg1=arg1,...,argN=argN)", must have "@app.route('/get_image_url/<argN>') decorator for each argument argN
@app.route('/get_image_url')
@app.route('/get_image_url/<filename>')
def get_image_url(filename):
    '''for use in <somefile>.html, so can call <img src={{ url_for('upload_send_image_url',filename=filename }}>'''
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)
#     return send_from_directory(app.config['UPLOAD_FOLDER'],
#                                "engage_1ps.jpg")

# class HowAnalyze(Form):
#     choices= [("1","Blob"),('2','HOG'),('3',"Blob and HOG")]
#     func= SelectMultipleField("HowAnalyze",choices=choices)
#     choices= [('1','basic'),('2','detailed')]
#     output= SelectField("what output",choices=choices)


#pure html to show uploaded image and html for for user select how analyze image
@app.route('/upload/analyze',methods=['GET','POST'])
@app.route('/upload/analyze/<filename>',methods=['GET','POST'])
def upload_analyze(filename):
#     if request.method == 'POST':
#         return "hello"
#         username= request.form.username
#         return redirect( url_for('get_image_url',filename=filename) )
    print "filename= %s" % filename
    return render_template('upload.html',filename=filename)

@app.route('/upload/analyze_CalcStuff',methods=['GET','POST'])
def upload_analyze_CalcStuff():
    if request.method == 'POST':
        print request.form.keys()
        for key, val in request.form.iteritems():
            print key, val
        if request.form['HowAnalyze'] == "Blob":
            return redirect( url_for('Blob_results',filename=request.form["filename"]))
        elif request.form['HowAnalyze'] == "HOG":
            return redirect( url_for('Hog_results',filename=request.form["filename"]))
        else: 
            return redirect( url_for('Blob_Hog_results',filename=request.form["filename"]))

@app.route('/get_tmp_image_url')
@app.route('/get_tmp_image_url/<filename>')
def get_tmp_image_url(filename):
    '''gets called from html files, with tag <img src={{}}'''
    return send_from_directory('tmp/',filename)

# @app.route('/upload/analyze/ML')
@app.route('/upload/analyze/Blob_results/<filename>')
def Blob_results(filename):
    #insert if, elif, else to show appropriate html file depending if HOG, Blob, or bothredirect to html first, print loading, then call functions then call html again but with results
    file=os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image= mpimg.imread(file)
    import Blob_Features as BF
    blob_stats= BF.BlobMethod_on_Image(image,file)
    blob_fig= "BlobFeaturesPlot.png"
    # print "blob_results.HasPeople, blob_results.frac_correct, blob_results.precision, blob_results.recall, blob_results.tp_norm, blob_results.fp_norm %s %f %f %f %f %f" % \
#     (blob_results.HasPeople, blob_results.frac_correct, blob_results.precision, blob_results.recall, blob_results.tp_norm, blob_results.fp_norm)
    return render_template('hog_blob_results.html',\
            blob_fig=blob_fig,blob_predict=blob_stats.HasPeople,blob_stats=blob_stats)

@app.route('/upload/analyze/Hog_results/<filename>')
def Hog_results(filename):
    #load image into array
    file=os.path.join(app.config['UPLOAD_FOLDER'], filename)
    obj = Image.open( file )
    rgb_img= np.array(obj)
    grey_img = np.array(obj.convert('L'))
    #hog ML
    sys.path.append('machine_learn/HOG/')
    import predict_on_UploadImage as HogML
    (et,svc)=HogML.Hog_predict_UploadImage(grey_img,file,rgb_img)
    et_ans= HogML.interpret_int_predict(et[0].astype('int'))
    svc_ans= HogML.interpret_int_predict(svc[0].astype('int'))
    hog_fig="image_of_hog.png"
    #get hog stats
    fin=open('machine_learn/HOG/hog_stats_10.pickle',"r")
    hog_stats=pickle.load(fin)
    fin.close()
    return render_template('hog_blob_results.html',\
            hog_fig=hog_fig,hog_predict=[et_ans,svc_ans],hog_stats=hog_stats)
    
@app.route('/upload/analyze/Blob_Hog_results/<filename>')
def Blob_Hog_results(filename):
    file=os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #HOG
    obj = Image.open( file )
    rgb_img= np.array(obj)
    grey_img = np.array(obj.convert('L'))
    sys.path.append('machine_learn/HOG/')
    import predict_on_UploadImage as HogML
    (et,svc)=HogML.Hog_predict_UploadImage(grey_img,file,rgb_img)
    et_ans= HogML.interpret_int_predict(et[0].astype('int'))
    svc_ans= HogML.interpret_int_predict(svc[0].astype('int'))
    hog_fig="image_of_hog.png"
    fin=open('machine_learn/HOG/hog_stats_10.pickle',"r")
    hog_stats=pickle.load(fin)
    fin.close()
    #BLOB
    image= mpimg.imread(file)
    import Blob_Features as BF
    blob_stats= BF.BlobMethod_on_Image(image,file)
    blob_fig= "BlobFeaturesPlot.png"
    #load html with both Hog and Blob vars
    return render_template('hog_blob_results.html',\
            hog_fig=hog_fig,hog_predict=[et_ans,svc_ans],hog_stats=hog_stats,\
            blob_fig=blob_fig,blob_predict=blob_stats.HasPeople,blob_stats=blob_stats)

#######################
    
if __name__ == '__main__':
    app.run(debug=True)
     #   host="0.0.0.0",
     #   port=int("80"),
     #   debug=True
    # 



