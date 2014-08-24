from flask import render_template, flash, redirect
from app import app

@app.route('/')
@app.route('/index', methods = ['GET', 'POST'])
def index():
    return render_template('index.html',
        title = 'Skin vs. Food')



