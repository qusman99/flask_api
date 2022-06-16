# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 08:22:38 2022

@author: qusma
"""



import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)


@app.route('/')
def hello():
    return "Welcome All to Week-14"



@app.route('/predict_test', methods=["POST"])
def predict_test_class():
    
    """Let's predict the class for Breast Cancer
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return " The Predicated Class for the TestFile is"+ str(list(prediction))


if __name__=='__main__':
    app.run()