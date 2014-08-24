from flask import render_template, flash, redirect
from app import app

from flask.ext.wtf import Form
from wtforms import RadioField,TextField, BooleanField, validators, TextAreaField, SubmitField
from werkzeug import secure_filename
from flask_wtf.file import FileField

from flaskext.uploads import UploadSet, IMAGES

class LoginForm(Form):
    openid = TextField('openid', [validators.Required()]) 
    remember_me = BooleanField('remember_me', default = False)
    textArea= TextAreaField("textarea",[validators.Required()])
    submit= SubmitField("submit",[validators.Optional()])

class UploadForm(Form):
    rgb_image = FileField('rgb_image')
    image_name = TextField('image_name', [validators.Required()]) 
#     choices=[(1,"Limbs"),(2,"Faces"),(3,"Food"),(4,"None of the above")] #list of tuples for RadioField
#     WhatSee= RadioField("WhatSee",choices=choices)#,[validators.Required()])
#     Notes= TextAreaField("Notes",[validators.Required()])
    submit= SubmitField("upload",[validators.Optional()])



@app.route('/')
@app.route('/index', methods = ['GET', 'POST'])
def index():
    return render_template('index.html',
        title = 'Skin vs. Food')

photos = UploadSet('photos', IMAGES)

@app.route('/canned_upload', methods=['GET', 'POST'])
def canned_upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        rec = Photo(filename=filename, user=g.user.id)
        rec.store()
        flash("Photo saved.")
        return redirect(url_for('show', id=rec.id))
    return render_template('canned_upload.html')

@app.route('/photo/<id>')
def show(id):
    photo = Photo.load(id)
    if photo is None:
        abort(404)
    url = photos.url(photo.filename)
    return render_template('show.html', url=url, photo=photo)

@app.route('/upload', methods = ['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        filename = secure_filename(form.rgb_image.data.filename)
        form.photo.data.save('uploads/' + filename)
#         flash('uploaded image: %s, User saw these in image: %s, submit= %r' %(form.image_name.data, form.WhatSee.data, str(form.submit.data)))
        flash('image name= %s' % form.image_name.data)
    else:
        filename=None
#         return redirect('/index')
    return render_template('upload.html',title = 'Skin vs. Food',form= form, filename=filename)

@app.route('/login', methods = ['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash('Login requested for OpenID="' + form.openid.data + '", remember_me=' + str(form.remember_me.data)+"  textarea="+form.textArea.data+"submit="+str(form.submit.data))
        return redirect('/index')
    return render_template('login.html', 
        title = 'Sign In',
        form = form,
        providers = app.config['OPENID_PROVIDERS'])

