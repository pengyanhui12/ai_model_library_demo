import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException

from inference import ONNXDetector


app = FastAPI(title="AI Model Library Demo")

MODEL_ROOT = Path("model_library")
MODEL_REGISTRY = {
    "yolo_demo": "vision/object_detection/yolo_demo"
}

loaded_models = {}


def get_model(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

    if model_name not in loaded_models:
        model_dir = MODEL_ROOT / MODEL_REGISTRY[model_name]
        loaded_models[model_name] = ONNXDetector(str(model_dir))

    return loaded_models[model_name]


@app.get("/")
def root():
    return {
        "message": "AI Model Library Demo is running",
        "available_models": list(MODEL_REGISTRY.keys())
    }

@app.get("/models")
def list_models():
    models = []

    for name, rel_path in MODEL_REGISTRY.items():
        config_path = MODEL_ROOT / rel_path / "config.json"

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}

        models.append({
            "model_name": name,
            "path": rel_path,
            "task": config.get("task"),
            "input_size": config.get("input_size")
        })

    return {
        "num_models": len(models),
        "models": models
    }


@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):
    model = get_model(model_name)

    content = await file.read()
    image_array = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = model.predict_image(image)
    return result