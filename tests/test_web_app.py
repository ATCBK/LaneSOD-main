from io import BytesIO

import pytest

from app import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "UPLOAD_DIR": tmp_path / "uploads",
            "RESULT_DIR": tmp_path / "results",
        }
    )
    app.config["PREDICT_IMAGE"] = lambda file_storage: {
        "result_url": "/static/generated/result.png"
    }

    with app.test_client() as test_client:
        yield test_client


def test_homepage_renders_lane_project_content(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "车道线分割".encode("utf-8") in response.data
    assert b"InSPyReNet" in response.data
    assert "在线演示".encode("utf-8") in response.data


def test_predict_requires_uploaded_file(client):
    response = client.post("/predict", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.get_json()["error"] == "请先选择图片文件。"


def test_predict_returns_result_url(client):
    response = client.post(
        "/predict",
        data={"image": (BytesIO(b"fake-image"), "lane.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["result_url"] == "/static/generated/result.png"
