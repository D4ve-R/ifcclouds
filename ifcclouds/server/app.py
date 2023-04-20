import os
import torch
import torch.nn as nn
from flask import Flask, url_for, request, jsonify
from dotenv import load_dotenv, find_dotenv


from ifcclouds.models.dgcnn import DGCNN_semseg

NUM_CLASSES = 9

app = Flask(__name__)
model = DGCNN_semseg(NUM_CLASSES)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'model.pt')))

def predict(input):
    model.eval()
    input_tensor = torch.tensor([input]).float()
    output = model(input_tensor)
    return output.tolist()

@app.route("/")
def index():
    # return index.html
    return "Hello World"

@app.route('/predict', methods=['POST'])
def handle_request():
    input = request.json['input']
    output = predict(input)
    return jsonify({'output': output})

if __name__ == "__main__":
    load_dotenv(find_dotenv('.flaskenv'))
    host = os.getenv('HOST')
    port = os.getenv('PORT')
    app.run(host, port)