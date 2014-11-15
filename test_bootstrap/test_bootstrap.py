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
    return render_template('index.html')

@app.route('/AboutUs')
def AboutUs():
    print "I am here:"
    os.system("pwd")
    return render_template('About_us.html')


##### main
    
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
#     app.run(debug=True)
     #   host="0.0.0.0",
     #   port=int("80"),
     #   debug=True
    # 
    



