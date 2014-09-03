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

class HowAnalyze(Form):
    choices= [("1","Blob"),('2','HOG'),('3',"Blob and HOG")]
    func= SelectMultipleField("HowAnalyze",choices=choices)
    choices= [('1','basic'),('2','detailed')]
    output= SelectField("what output",choices=choices)

#flask wtforms to show uploaded image and html for for user select how analyze image
# @app.route('/upload/analyze')
# @app.route('/upload/analyze/<filename>')
# def upload_analyze(filename):
#     form= HowAnalyze()
#     if form.validate_on_submit():
#         # flash('User Sees: %s, test= %s' % \
# #             (form.UserSees.data))
#         return redirect( url_for('get_image_url',filename) )
#     return render_template('upload.html',filename=filename,form=form)
# 
#        <form action="" method="post" name="temp">
#             {{form.hidden_tag()}}
#             <p><label for="title">{{form.func.label}}</label><br/>
#                 {{form.func}}
#             </p>
#             <p><label for="title">{{form.output.label}}</label><br/>
#                 {{form.output}}
#             </p>
#             <p><input type="submit" name='Analyze' value="Analyze"></p>
#         </form>

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
            return redirect( url_for('Blob_only',filename=request.form["filename"]))
        elif request.form['HowAnalyze'] == "HOG":
            return redirect( url_for('HOG_only',filename=request.form["filename"]))
        else: return "neither Blob nor HOG alone"

# @app.route('/upload/analyze/ML')
@app.route('/upload/analyze/ML/<filename>')
def Blob_only(filename):
    #insert if, elif, else to show appropriate html file depending if HOG, Blob, or bothredirect to html first, print loading, then call functions then call html again but with results
    file=os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image= mpimg.imread(file)
    import Blob_Features as BF
    blob_results= BF.BlobMethod_on_Image(image,file)
    print "blob_results.HasPeople, blob_results.frac_correct, blob_results.precision, blob_results.recall, blob_results.tp_norm, blob_results.fp_norm %s %f %f %f %f %f" % \
    (blob_results.HasPeople, blob_results.frac_correct, blob_results.precision, blob_results.recall, blob_results.tp_norm, blob_results.fp_norm)
    return "Blob only"#render_template('Blob_and_HOG_results.html',filename=filename,\
                #blob_results=blob_results)

@app.route('/upload/analyze/HOG_only/<filename>')
def HOG_only(filename):
    file=os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image= mpimg.imread(file)
    return "HOG_only"
  
#######################
    
if __name__ == '__main__':
    app.run(debug=True)
     #   host="0.0.0.0",
     #   port=int("80"),
     #   debug=True
    # 



