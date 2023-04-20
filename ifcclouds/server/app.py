import os
from flask import Flask, url_for
from dotenv import load_dotenv, find_dotenv

app = Flask(__name__)
@app.route("/")
def index():
    # return index.html
    return "Hello World"

@app.route("/convert")
def convert():
    # return convert.html
    return "Convert"

if __name__ == "__main__":
    load_dotenv(find_dotenv('.flaskenv'))
    host = os.getenv('HOST')
    port = os.getenv('PORT')
    app.run(host, port)