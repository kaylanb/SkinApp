from flask import Flask
from flask import render_template, flash, redirect
import numpy as np
#forms
from flask.ext.wtf import Form
from wtforms import TextField, SubmitField, TextAreaField
from wtforms.validators import Required, Optional
#other
from pandas import read_csv, DataFrame

app = Flask(__name__)
app.config.from_object('config')

class Click2Play(Form):
    type=TextField('Anything', validators = [Required()])
    play= SubmitField("Click2Play")

class getWhatUserSees(Form):
    UserSees = TextField('UserSees', validators = [Required()])
    test= TextAreaField("test",validators= [Optional()])
    MlOnImage= SubmitField("MachineLearnOnImage",validators = [Required()])
   
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
    
if __name__ == '__main__':
    app.run(debug=True)
     #   host="0.0.0.0",
     #   port=int("80"),
     #   debug=True
    # 



