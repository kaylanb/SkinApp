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


app = Flask(__name__)
app.config.from_object('config')

class Click2Play(Form):
    type=TextField('Anything', validators = [Required()])
    play= SubmitField("Click2Play")

class getWhatUserSees(Form):
    UserSees = TextField('UserSees', validators = [Required()])
    test= TextAreaField("test",validators= [Optional()])
    MlOnImage= SubmitField("MachineLearnOnImage",validators = [Required()])
   
class SelectOption(Form):
    choices= [("1","Use Test Image"),('2','Upload my own image')]
    HowAnalyze= SelectField("HowAnalyze",choices=choices)
    choices= [('1','Blob'),('2','HOG'),('3','Blob features'),('4','HOG features')]
    WhichMethods= SelectMultipleField("Which Methods",choices=choices)

class UploadAnalyze(Form):
    choices= [('1','Blob'),('2','HOG'),('3','Blob features'),('4','HOG features')]
    HowAnalyze= SelectField("HowAnalyze",choices=choices)

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
#     form= Click2Play()
    form= SelectOption()
    if form.validate_on_submit():
        flash('Login requested for OpenID="' + form.openid.data + '", remember_me=' + str(form.remember_me.data))
        return redirect('/AnalyzeImg')
    return render_template('index.html',title = 'Select Option',form=form)

@app.route('/AnalyzeImg', methods = ['GET', 'POST'])
def AnalyzeImg():
    form= getWhatUserSees()
    path="training_image_urls/"
    urlFile= "NewTraining_Food_everyones.txt"
    urls = np.loadtxt(path+ urlFile, dtype="str")
    if form.validate_on_submit():
        flash("HERE I AM")
        flash('User Sees: %s, test= %s' % \
            (form.UserSees.data))
        return redirect('/ML')#,img_url=urls[0])
    return render_template('AnalyzeImg.html',
        title = 'AnalyzeImg',urls=urls,form=form)

# @app.route('/OutputResults', methods = ['GET', 'POST'])
# def OutputResults():
#     do_ML()


@app.route('/ML', methods = ['GET', 'POST'])#img_url=image_url)
def ML():
    f=open("tmp/blob.txt","r")
    blob_txt= f.read()
    f.close()
    f=open("tmp/hog.txt","r")
    hog_txt= f.read()
    f.close()
#     urls=request.args.get('urls')
#     flash(urls[0] )
    #fig out how pass: UserSaw=form.UserSees.data and image_url into this function!\
    return render_template('ML.html',title = 'ML Results', \
            blob_image_results="tmp/blob.png",blob_text_results=blob_txt, \
            hog_image_results="tmp/hog.png",hog_text_results=hog_txt)

#################### upload file code
# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/upload')
def upload_start_pt():
    return render_template('upload.html')


class TestRedirectForm(Form):
    remember_me = BooleanField('remember_me', default = False)

# Route that will process the file upload
@app.route('/upload/savefile', methods=['GET','POST'])
def upload():
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
#         return redirect(url_for('uploaded_file',
#                                 filename=filename))
        return redirect(url_for('func_call_analyze'))
        # image_url=url_for('uploaded_file',filename=filename)
#         form =TestRedirectForm()
#         if request.method == 'POST':
#             return redirect('/analyze')#,image_url=image_url,saved_at=saved_at))
#         return render_template('upload_analyze.html',image_url=image_url,saved_at=saved_at,form=form)
        

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
# @app.route('/uploads/<filename>')
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/analyze', methods = ['GET', 'POST'])
def func_call_analyze():#image_url,saved_at):
    return "successfully redirected to /analyze!"
    # form = UploadAnalyze()
#     if form.validate_on_submit():
#         flash('User uploaded their own image and chose to analyze with: ' + "IAMHERE")#form.HowAnalyze.data )
#         return redirect('/')
#     return render_template('upload_analyze.html',image_url=image_url,saved_at=saved_at,form=form)
    
#######################
    
if __name__ == '__main__':
    app.run(debug=True)
     #   host="0.0.0.0",
     #   port=int("80"),
     #   debug=True
    # 



