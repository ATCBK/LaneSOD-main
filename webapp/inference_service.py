from importlib import import_module
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from werkzeug.utils import secure_filename

from utils.custom_transforms import *  # noqa: F401,F403
from utils.dataloader import *  # noqa: F401,F403
from utils.utils import load_config, move_sample_to_device, resolve_device


MODEL_IMPORTS = {
    "InSPyReNet_Res2Net50": ("lib.InSPyReNet_Res2Net50", "InSPyReNet_Res2Net50"),
    "InSPyReNet_SwinB": ("lib.InSPyReNet_SwinB", "InSPyReNet_SwinB"),
}


class LaneWebPredictor:
    def __init__(
        self,
        upload_dir,
        result_dir,
        config_path="configs/HighwayLane.yaml",
        mask_path="data/mask.png",
        public_result_prefix="/static/generated",
        output_type="overlay",
    ):
        self.upload_dir = Path(upload_dir)
        self.result_dir = Path(result_dir)
        self.config_path = config_path
        self.mask_path = Path(mask_path)
        self.public_result_prefix = public_result_prefix.rstrip("/")
        self.output_type = output_type

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.device = resolve_device()
        self._config = None
        self._model = None
        self._transform = None
        self._mask = None

    def predict(self, file_storage):
        original_name = secure_filename(file_storage.filename or "upload.png")
        stem = Path(original_name).stem or "upload"
        suffix = Path(original_name).suffix or ".png"
        token = uuid4().hex

        upload_path = self.upload_dir / f"{token}-{stem}{suffix}"
        output_path = self.result_dir / f"{token}-{stem}.png"

        file_storage.save(upload_path)
        self._run_prediction(upload_path, output_path)

        return {"result_url": f"{self.public_result_prefix}/{output_path.name}"}

    def predict_path(self, image_path, output_path):
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._run_prediction(image_path, output_path)
        return output_path

    def _load_runtime(self):
        if self._model is not None:
            return

        self._config = load_config(self.config_path)
        module_name, class_name = MODEL_IMPORTS[self._config.Model.name]
        model_cls = getattr(import_module(module_name), class_name)
        self._model = model_cls(
            channels=self._config.Model.channels,
            pretrained=self._config.Model.pretrained,
        )

        checkpoint_path = Path(self._config.Test.Checkpoint.checkpoint_dir) / "latest.pth"
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self._model.load_state_dict(state_dict, strict=True)
        self._model.to(self.device)
        self._model.eval()

        dataset_type = eval(self._config.Test.Dataset.type)
        self._transform = dataset_type.get_transform(self._config.Test.Dataset.transform_list)

        if self.mask_path.exists():
            self._mask = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            self._mask = None

    def _run_prediction(self, image_path, output_path):
        self._load_runtime()

        image = Image.open(image_path).convert("RGB")
        sample = {"image": image}
        sample = self._transform(sample)
        sample["image"] = sample["image"].unsqueeze(0)
        sample = move_sample_to_device(sample, self.device)

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda",
            ):
                output = self._model(sample)

        prediction = F.interpolate(
            output["pred"],
            image.size[::-1],
            mode="bilinear",
            align_corners=True,
        )
        prediction = torch.sigmoid(prediction).detach().cpu().numpy().squeeze()
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
        prediction = (prediction * 255).astype(np.uint8)

        if self._mask is not None and self._mask.shape == prediction.shape:
            prediction = prediction * (self._mask != 0)

        result_image = self._build_result_image(np.array(image), prediction)
        Image.fromarray(result_image).save(output_path)

    def _build_result_image(self, original_rgb, prediction):
        if self.output_type == "map":
            return prediction

        if self.output_type == "rgba":
            r, g, b = cv2.split(original_rgb)
            return cv2.merge([r, g, b, prediction])

        prediction_float = prediction.astype(np.float32) / 255.0
        prediction_float[prediction_float < 0.5] = 0
        alpha = np.expand_dims(prediction_float, axis=-1)
        overlay_color = np.array([255, 0, 0], dtype=np.float32)
        return (
            original_rgb.astype(np.float32) * (1 - alpha * 0.6)
            + overlay_color * (alpha * 0.6)
        ).astype(np.uint8)
