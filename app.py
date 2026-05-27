import pickle

from flask import Flask,request, jsonify, render_template, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

ridge_model = pickle.load(open('notebook/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('notebook/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])
        input_data = np.array([[temperature, RH,Ws, rain, FFMC, DMC, ISI, Classes, Region]])
        input_data_scaled = standard_scaler.transform(input_data)
        prediction = ridge_model.predict(input_data_scaled)
        return render_template('home.html', result='Predicted Fire Size: {:.2f}'.format(prediction[0]))
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)