#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:55:22 2023

@author: elsayed
"""
#Import Liberaries
import numpy as np
from flask import Flask, request, jsonify, render_template
import json 
import re 
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import load_model


#Define App
app = Flask(__name__, template_folder='templates')
#an object of class Flask which would do all the heavy lifting for us, like handling the incoming requests from the browser 
# and providing with appropriate responses (in this case our model prediction) in some nice form using html,css. 

#load Model
model=load_model('SaveModels\.ipynb_checkpoints\model.h5') # load model 


with open('intents.json') as file:
    data=json.load(file)

    

@app.route('/')
def home():
    return render_template('home1.html')


@app.route('/',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    value=request.form.get("user")
    toknizer=pickle.load(open(r'SaveModels\.ipynb_checkpoints\toknizer.pkl','rb'))
    le=pickle.load(open(r'SaveModels\.ipynb_checkpoints\label_encoder.pkl','rb'))
    sent=pad_sequences(toknizer.texts_to_sequences([value]),maxlen=20) #[] len=20

    result=np.argmax(model.predict(np.array(sent))) # 0-7 
    # 2,3 , 4
    f_res=le.inverse_transform(np.array(result).reshape(1))

    output=''
    for label in data['intents']:

        if label['tag']==f_res:

            output=np.random.choice(label['responses'])



    return render_template('home1.html', prediction_text='{} '.format(output))



if __name__ == "__main__":
    app.run(debug=True)
