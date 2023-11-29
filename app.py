import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pd.read_pickle(r'C:\Users\DELL\OneDrive\Desktop\ML Model\new_model.pickle')

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The insurance is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)