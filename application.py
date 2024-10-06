from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # This is correct

application = Flask(__name__)
app = application

# Load Ridge regression model and Standard Scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/hello")
def hello():
    return "<H1>Hello there<H1>"

@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/predict",methods=['POST','GET'])
def predict():  
    if request.method=='POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))  # Assuming this is a string
        Region = float(request.form.get('Region')) # Assuming this is a string

        new_scaled=standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result=ridge_model.predict(new_scaled)
        return render_template('home.html',results=result)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
