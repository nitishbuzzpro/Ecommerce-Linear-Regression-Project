#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from flask import Flask, render_template, request
import pickle
import numpy as np


# Construct the absolute file path to the pickle file on PythonAnywhere
pickle_file_path = "Ecommerce-Linear Regression.pkl"

# Load the machine learning model from the pickle file
with open(pickle_file_path, 'rb') as f:
    model = pickle.load(f)

model = pickle.load(open("Ecommerce-Linear Regression.pkl",'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_amount_spent():
    Avg_Session_Length = float(request.form.get('Avg_Session_Length'))
    Time_on_App = float(request.form.get('Time_on_App'))
    Time_on_Website = float(request.form.get('Time_on_Website'))
    Length_of_Membership = float(request.form.get('Length_of_Membership'))

    result = model.predict(np.array([Avg_Session_Length,Time_on_App,Time_on_Website,Length_of_Membership]).reshape(1,4))

    if result < 0:
        result = str(0)
        return render_template('index.html',result=result)
    else:
        result = str(round(result[0],2))
        return render_template('index.html',result=result)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


# In[ ]:




