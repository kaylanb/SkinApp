from flask import Flask
from flask import render_template, flash, redirect
import numpy as np
#forms
# from flask.ext.wtf import Form
# from wtforms import TextField, SubmitField, TextAreaField, BooleanField, SelectField,SelectMultipleField
# from wtforms.validators import Required, Optional
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
import os


app = Flask(__name__)
app.config.from_object('config')

@app.route('/', methods=['GET','POST'])
# @app.route('/index', methods = ['GET', 'POST'])
def index():
    return render_template('index_bootstrap.html')#('index.html')

@app.route('/OurTeam')
def OurTeam():
    print "I am here:"
    os.system("pwd")
    return render_template('our_team.html')

@app.route('/AboutTheProject')
def AboutProject():
    print "I am here:"
    os.system("pwd")
    return render_template('about_the_project.html')

# @app.route('/BootStrap')
# def BootStrap():
#     return render_template('index_bootstrap.html')


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

#convert prediction integer to meaningful string: 0 -> food, 1 -> people
def int_predict(int_predict):
    food_code=0
    people_code=1
    if int_predict == food_code: return "Food"
    else: return "People"

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
        ans_pred_d["ans"]= int_predict( blob_df.answer.values[index].astype('int') )
        ans_pred_d["blob_pred"]= int_predict( blob_df.predict.values[index].astype('int') )
        index= np.where(hog_df.url == url)[0][0]
        ans_pred_d["hog_pred_et"]= int_predict(hog_df.ET_predict.values[index].astype('int'))
        ans_pred_d["hog_pred_svc"]= int_predict(hog_df.LinSVC_predict.values[index].astype('int'))
        #blob stats
        fin=open(root+'NoLims_shuffled_blob_stats.pickle')
        blob_stats=pickle.load(fin)
        fin.close()
        #hog stats from 10 Trials through Train/Test set
        fin=open('machine_learn/HOG/hog_stats_10.pickle',"r")
        hog_stats=pickle.load(fin)
        fin.close()
        print "answer type is: ", type(ans_pred_d["ans"])
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

##Upload image code:

#upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# extensions allowed for uploaded image
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
        #show the file and ask for what user sees
        save_to_tmp(filename,"upload_filename.pickle")
        return render_template('upload_show.html',filename=filename) 

@app.route('/upload_process/blob_hog_results', methods=['POST'])
def upload_blob_hog_results():
    if request.method == 'POST':  
        #if here then user entered what saw
        if request.method == 'POST' and request.form["submit"]=="restart":  
            #if here then user said do again for new image
            return redirect('/upload') #url_for('upload_analyze'))
        filename= load_from_tmp("upload_filename.pickle")
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
        return render_template('hog_blob_results.html',filename=filename,\
    WhatSee=request.form["WhatSee"],\
    hog_fig=hog_fig,hog_predict=[et_ans,svc_ans],hog_stats=hog_stats,\
    blob_fig=blob_fig,blob_predict=blob_stats.HasPeople,blob_stats=blob_stats)
    else: return "got here by error!"
       
#return path to images in /uploads
@app.route('/get_image_url')
@app.route('/get_image_url/<filename>')
def get_image_url(filename):
    '''for use in <somefile>.html, so can call <img src={{ url_for('upload_send_image_url',filename=filename }}>'''
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

#return path to images in /tmp
@app.route('/get_tmp_image_url')
@app.route('/get_tmp_image_url/<filename>')
def get_tmp_image_url(filename):
    '''gets called from html files, with tag <img src={{}}'''
    return send_from_directory('tmp/',filename)


##### main
    
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
#     app.run(debug=True)
     #   host="0.0.0.0",
     #   port=int("80"),
     #   debug=True
    # 
    



