# Lane Web Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal Flask-based web demo for the lane segmentation project with a real single-image inference flow that runs on the current CPU-based machine.

**Architecture:** Add a small Flask application around the existing repository, extract inference into a reusable service layer, and convert CUDA-only paths to device-aware logic. Keep the frontend as a single presentation page with one real upload interaction.

**Tech Stack:** Python, Flask, PyTorch, HTML, CSS, vanilla JavaScript, pytest

---

### Task 1: Add Web App Test Coverage

**Files:**
- Create: `tests/test_web_app.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write the failing test**

```python
from io import BytesIO


def test_homepage_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Lane Segmentation" in response.data


def test_predict_returns_result_url(client):
    response = client.post(
        "/predict",
        data={"image": (BytesIO(b"fake-image"), "lane.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert "result_url" in payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_app.py -v`
Expected: FAIL because the Flask app module and endpoint do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from flask import Flask, jsonify

app = Flask(__name__)


@app.get("/")
def index():
    return "Lane Segmentation"


@app.post("/predict")
def predict():
    return jsonify({"result_url": "/static/results/demo.png"})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_web_app.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_web_app.py requirements.txt
git commit -m "test: add web app endpoint coverage"
```

### Task 2: Make Inference CPU/GPU Safe

**Files:**
- Create: `webapp/inference_service.py`
- Modify: `run/Inference.py`
- Modify: `utils/utils.py`
- Modify: `lib/modules/layers.py`
- Test: `tests/test_web_app.py`

- [ ] **Step 1: Write the failing test**

```python
def test_predict_uses_inference_service_result(client, monkeypatch, tmp_path):
    def fake_predict(file_storage):
        output = tmp_path / "result.png"
        output.write_bytes(b"png")
        return {"result_url": "/static/results/result.png"}

    monkeypatch.setattr("app.predict_image", fake_predict)

    response = client.post(
        "/predict",
        data={"image": (BytesIO(b"image"), "lane.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json()["result_url"] == "/static/results/result.png"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_app.py::test_predict_uses_inference_service_result -v`
Expected: FAIL because the app does not delegate to a real service.

- [ ] **Step 3: Write minimal implementation**

```python
def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_sample_to_device(sample, device):
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            sample[key] = value.to(device)
    return sample
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_web_app.py::test_predict_uses_inference_service_result -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add webapp/inference_service.py run/Inference.py utils/utils.py lib/modules/layers.py tests/test_web_app.py
git commit -m "feat: add device-safe inference service"
```

### Task 3: Build the Demo UI

**Files:**
- Create: `templates/index.html`
- Create: `static/styles.css`
- Create: `static/app.js`
- Modify: `app.py`
- Test: `tests/test_web_app.py`

- [ ] **Step 1: Write the failing test**

```python
def test_homepage_contains_demo_sections(client):
    response = client.get("/")
    assert b"Project Overview" in response.data
    assert b"Run Live Demo" in response.data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_app.py::test_homepage_contains_demo_sections -v`
Expected: FAIL because the template does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```html
<section>
  <h2>Project Overview</h2>
</section>
<section>
  <h2>Run Live Demo</h2>
</section>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_web_app.py::test_homepage_contains_demo_sections -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add templates/index.html static/styles.css static/app.js app.py tests/test_web_app.py
git commit -m "feat: add lane demo landing page"
```

### Task 4: End-to-End Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the failing test**

```python
def test_upload_requires_a_file(client):
    response = client.post("/predict", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_app.py::test_upload_requires_a_file -v`
Expected: FAIL until request validation is added.

- [ ] **Step 3: Write minimal implementation**

```python
if "image" not in request.files or request.files["image"].filename == "":
    return jsonify({"error": "Please choose an image file."}), 400
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_web_app.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md app.py tests/test_web_app.py
git commit -m "docs: document lane demo web app usage"
```
