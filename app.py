from flask import Flask, render_template, request
import pandas as pd

from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "area": [float(request.form["area"])],
        "bedrooms": [int(request.form["bedrooms"])],
        "bathrooms": [int(request.form["bathrooms"])],
        "stories": [int(request.form["stories"])],
        "mainroad": [request.form["mainroad"]],
        "guestroom": [request.form["guestroom"]],
        "basement": [request.form["basement"]],
        "hotwaterheating": [request.form["hotwaterheating"]],
        "airconditioning": [request.form["airconditioning"]],
        "parking": [int(request.form["parking"])],
        "prefarea": [request.form["prefarea"]],
        "furnishingstatus": [request.form["furnishingstatus"]]
    }

    input_df = pd.DataFrame(data)

    pipeline = PredictionPipeline()

    prediction = pipeline.predict(input_df)

    return render_template(
        "result.html",
        prediction=f"₹ {prediction[0]:,.2f}",
        model_name="Linear Regression",
        accuracy="85.29%",
        features=12
    )


if __name__ == "__main__":
    app.run(debug=True)