import os
import torch
import torch.nn as nn
from flask import Flask, url_for, request, jsonify, redirect, render_template, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv, find_dotenv

from ifcclouds.models.dgcnn import DGCNN_semseg
from ifcclouds.convert import process_ifc
from ifcclouds.data.dataset import default_classes

NUM_CLASSES = len(default_classes)

app = Flask(__name__)
model = DGCNN_semseg(NUM_CLASSES)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'model.pt'), map_location=torch.device('cpu')))

def predict(input):
    model.eval()
    input_tensor = torch.tensor([input]).float()
    output = model(input_tensor)
    output = output.permute(0, 2, 1).contiguous()
    output = output.max(dim=2)[1]
    return output.tolist()

@app.route("/")
def index():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    processed_dir = os.path.join(data_dir, 'processed')
    raw_dir = os.path.join(data_dir, 'raw')
    processed_files = [f for f in os.listdir(processed_dir) if os.path.isfile(os.path.join(processed_dir, f)) and f.endswith('.ply')]
    raw_files = [f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f)) and f.endswith('.ifc')]
    return render_template('index.html', processed_files=processed_files, raw_files=raw_files)

@app.route('/train', methods=['POST'])
def train():
    learning_rate = request.form['learning_rate']
    momentum = request.form['momentum']

@app.route('/predict', methods=['POST'])
def handle_request():
    input = request.json['input']
    output = predict(input)
    return jsonify({'output': output})

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', filename))
    file.save(file_path)
    process_ifc(file_path, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')))
    filename = filename.replace('.ifc', '.ply')
    return redirect(url_for('index'))

@app.route('/viz')
def viz():
    return render_template('viz.html')

@app.route('/file/<filename>')
def file(filename):
    return send_file(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', filename)))


if __name__ == "__main__":
    load_dotenv(find_dotenv('.flaskenv'))
    host = os.getenv('HOST')
    port = os.getenv('PORT')
    app.run(host, port)