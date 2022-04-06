# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 11:46:03 2022

@author: User
"""

import numpy as np
from flask import Flask, request, render_template
import pickle

# flask app
app = Flask(__name__,template_folder='templates')
# loading model
model = pickle.load(open('bmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict' ,methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = bmodel.predict(final_features)

 
    return render_template('home.html', output='Child status is :  {}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
    
    
    
    
    
    
    
    
    
