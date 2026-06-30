from flask import Flask, render_template, request
import pandas as pd

from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
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

        predicted_price = prediction[0]

        return render_template(
            "result.html",
            prediction=f"₹ {predicted_price:,.2f}",
            model_name="Linear Regression",
            accuracy="65.29%",
            features=12
        )

    except Exception as e:
        return f"<h2>Error:</h2><p>{str(e)}</p>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)