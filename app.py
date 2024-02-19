from flask import Flask,request, url_for, redirect, render_template
import tensorflow as tf
from transformers import AutoTokenizer
import numpy as np
import os


app = Flask(__name__)
model = tf.saved_model.load('saved_model/1')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')





@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        

        user_input = request.form['user_input']
        inputs = tokenizer(user_input, padding=True, truncation=True, return_tensors="tf")
        outputs = model(inputs)
        predictions = tf.argmax(outputs['logits'], axis=1).numpy()
        
        if predictions[0] == 0:
            result = 'The text is classified as "Normal".'
        elif predictions[0] == 1:
            result = 'The text is classified as "Attack".'
        else:
            result = 'An error occurred during the prediction.'

        return render_template('index.html',user_input = user_input,result=result)
    return render_template('index.html')
    



if __name__ == '__main__':
    app.run(debug=True)