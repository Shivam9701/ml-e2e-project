from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import PredictionPipeline, CustomData

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("home.html")

    data = request.form.to_dict()
    data = CustomData(**data).to_dataframe()

    pipeline = PredictionPipeline()
    prediction = pipeline.predict(data)

    return render_template("home.html", results=prediction[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0")
