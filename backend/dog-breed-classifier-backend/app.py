

from run_model import dog_breed_detector,init_model

from flask import Flask,request
from flask_cors import CORS
import json

app=Flask(__name__)
CORS(app)


model=init_model()

@app.route("/")
def init():
    return "<p>Hello, World!</p>"

@app.route("/identify_breed",methods=['POST'])
def identify_breed():
    if request.method == 'POST':
        f = request.form['image']
        breed_list=dog_breed_detector(f,model)
        return {
            'breed_list':json.dumps(breed_list)
        }

