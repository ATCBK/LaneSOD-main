from pathlib import Path

from flask import Flask, jsonify, render_template, request

from webapp import LaneWebPredictor


def create_app(test_config=None):
    app = Flask(__name__)

    base_dir = Path(__file__).resolve().parent
    app.config.update(
        UPLOAD_DIR=base_dir / ".uploads",
        RESULT_DIR=base_dir / "static" / "generated",
        PREDICT_IMAGE=None,
    )

    if test_config:
        app.config.update(test_config)

    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["RESULT_DIR"]).mkdir(parents=True, exist_ok=True)

    if app.config["PREDICT_IMAGE"] is None:
        predictor = LaneWebPredictor(
            upload_dir=app.config["UPLOAD_DIR"],
            result_dir=app.config["RESULT_DIR"],
        )
        app.config["PREDICT_IMAGE"] = predictor.predict

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        image = request.files.get("image")
        if image is None or image.filename == "":
            return jsonify({"error": "请先选择图片文件。"}), 400

        predictor = app.config["PREDICT_IMAGE"]
        if predictor is None:
            return jsonify({"error": "预测服务当前不可用。"}), 503

        return jsonify(predictor(image))

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
