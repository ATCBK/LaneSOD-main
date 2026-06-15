from io import BytesIO

from PIL import Image
from werkzeug.datastructures import FileStorage

from webapp.inference_service import LaneWebPredictor


def build_test_upload(filename="lane.png", size=(16, 16), color=(40, 60, 90)):
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    buffer.seek(0)
    return FileStorage(stream=buffer, filename=filename, content_type="image/png")


def test_predictor_saves_upload_and_result_files(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    result_dir = tmp_path / "results"
    predictor = LaneWebPredictor(upload_dir=upload_dir, result_dir=result_dir)

    def fake_run_prediction(image_path, output_path):
        Image.new("RGB", (16, 16), (255, 0, 0)).save(output_path)

    monkeypatch.setattr(predictor, "_run_prediction", fake_run_prediction)

    payload = predictor.predict(build_test_upload())

    assert payload["result_url"].startswith("/static/generated/")
    assert any(upload_dir.iterdir())
    assert any(result_dir.iterdir())
