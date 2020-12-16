import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle
import os
from os import listdir
import pandas as pd

path_now = os.getcwd()

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))

UPLOAD_FOLDER = path_now + "/uploadfile/"
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

@app.route('/')
#def home():
#    return render_template('index.html')
def home():
    return render_template('upload_csv.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route("/upload", methods=['POST'])
def uploadFiles():
        # get the uploaded file
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_path = app.config['UPLOAD_FOLDER']
            uploaded_file.save(file_path + uploaded_file.filename)\

        your_list = listdir(file_path)
        return render_template("upload_csv.html", your_list=your_list)
        # return redirect(url_for('home'))

import mymodel

@app.route("/train", methods=['POST'])
def trainmodel():
       
        namefile = request.form['namefile']
        try:
            file_path = app.config['UPLOAD_FOLDER']
            dataset = pd.read_csv(file_path + namefile)
        except:
            print("Error")
            return render_template("upload_csv.html")
        mymodel.train(dataset)
        return render_template("upload_csv.html")
        

@app.route("/test", methods=['POST'])
def mypredict():
        
        namefile = request.files['testfile']
        result = "None"
        score = "None"
        if namefile.filename != '':
            result,score = mymodel.test(namefile)
        return render_template("upload_csv.html",result=result,score=score)
      
    

if __name__ == "__main__":
    app.run(debug=True)