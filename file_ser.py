import flask

from flask_cors import CORS

app = flask.Flask(__name__, static_folder='./models')

CORS(app, resources=r'/*')

if __name__ == "__main__":
    app.run()