from flask import Flask,render_template,request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import jsonify
import essay_grader
import pandas as pd
import essay_trainer
import io
import os
import csv

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def recommend():
    fileInput = request.files['file'] 
    score = fileInput.read()
    score = str(score.strip())
    setNo = request.form['set']
    final_score = str(essay_grader.grade_score(str(setNo),score))
    return render_template("index.html",score=final_score)

@app.route('/trainer')
def trainer():
    return render_template("trainer.html")

@app.route('/trainer', methods=['POST'])
def trainEssays():
    f = request.files['fileTraining']
    f.save(secure_filename("trial25.csv"))
    data = pd.read_csv(('trial25.csv'), encoding='unicode_escape')
    essay_trainer.driver(data)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=False,threaded=False)    