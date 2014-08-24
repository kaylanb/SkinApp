from flask import render_template, flash, redirect
import numpy as np
from app import app
#forms
from flask.ext.wtf import Form
from wtforms import TextField, SubmitField, TextAreaField
from wtforms.validators import Required, Optional
#Algorithm table, ML results
import ML_results

class Click2Play(Form):
    type=TextField('Anything', validators = [Required()])
    play= SubmitField("Click2Play")

class getWhatUserSees(Form):
    UserSees = TextField('UserSees', validators = [Required()])
    test= TextAreaField("test",validators= [Optional()])
    MlOnImage= SubmitField("MachineLearnOnImage",validators = [Required()])
    

@app.route('/')
@app.route('/index', methods = ['GET', 'POST'])
def index():
    form= Click2Play()
    if form.validate_on_submit():
        return "I AM HERE"
#         flash("HERE")
#         return redirect(url_for('AnalyzeImg'))
    else: return render_template('index.html',title = 'Skin vs. Food',form=form)

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
    
    



