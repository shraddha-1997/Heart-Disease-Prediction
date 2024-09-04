from flask import Flask, render_template, request, redirect
import pickle
import numpy as np
import pandas as pd
import random

app=Flask(__name__)
#scaler=pickle.load(open('scaler.pkl','rb'))
model=pickle.load(open('classifier.pkl','rb'))
#model1=pickle.load(open('linear1.pkl', 'rb'))

@app.route('/')
def welcome_page():
    return render_template('welcomePage.htm')

@app.route('/home',methods=['GET','POST'])
def home():
    print('going home')
    return render_template('home.htm')


@app.route('/predict',methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    age = request.form['age']
    sex = request.form['sex']
    cp = request.form['cp']
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    exange = request.form['exange']
    oldpeak = request.form['oldpeak']
 
    age = int(age)
    sex = int(sex)
    cp = int(cp)
    trestbps = int(trestbps)
    chol = int(chol)
    fbs = int(fbs)
    restecg = float(restecg)
    thalach = float(thalach)
    exange = float(exange)
    oldpeak = float(oldpeak)

    
#   int_features = [int(x) for x in request.form.values()]
    final_features = np.array([(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exange,oldpeak)])
    sc=pickle.load(open('scaler.pkl','rb'))
    final_features=sc.transform(final_features)

    prediction = model.predict(final_features)
    #output = round(prediction[0], 2)
    return render_template("result.htm", pred = prediction)

if __name__=="__main__":
    app.run(debug=True)