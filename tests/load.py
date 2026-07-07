import os
from pathlib import Path

import numpy as np
import redis
import torch

from app.predictors.process_vits import ViTWrapper
from app.predictors.vector_similarity import VectorSimilarity

# Load secrets from .env
from dotenv import load_dotenv

load_dotenv()

test_images = Path(__file__).parent / "images" / "atolla"
DEFAULT_MODEL = "/models/vit-base-patch16-224"
DEFAULT_HOST = os.getenv("REDIS_HOST", "localhost")
vector_dimensions = 768  # Default for ViT base model
reset = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
r = redis.Redis(host=DEFAULT_HOST, port=6379, db=0, password=os.getenv("REDIS_PASSWD"))
vs = VectorSimilarity(r, vector_dimensions=vector_dimensions, reset=reset)
v = ViTWrapper(r, device, DEFAULT_MODEL)

for image_path in test_images.glob("*.png"):
    print(f"Processing image: {image_path}")
    inputs, _, _ = next(v.preprocess_images([image_path.as_posix()]))
    emb = v.get_image_embeddings(inputs)
    emb = emb.astype(np.float32)
    print(f"Loaded embedding for {image_path}: {emb.shape} to redis")
    id = image_path.stem
    vs.add_vector(doc_id=f"{image_path.stem}:{id}", vector=emb.tobytes(), tag=DEFAULT_MODEL)

# Grab all images in the test_images directory and predict
image_paths = list(test_images.glob("*.png"))
predictions, scores, ids = v.predict(image_paths, top_n=1)
print(f"Predictions: {predictions}, Scores: {scores}, IDs: {ids}")
