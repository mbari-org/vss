# fastapi-vss, Apache-2.0 license
# Filename: predictors/tasks.py
# Description: Worker task file for processing images with Vision Transformer (ViT) models
import gc
import json
import logging
import os
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import redis
from rq.local import LocalStack
from rq.worker import SimpleWorker

from app.config import init_config
from app.predictors.process_vits import ViTWrapper

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
_predictor_stack = LocalStack()


class MyWorker(SimpleWorker):
    def __init__(self, project, *args, **kwargs):
        super(MyWorker, self).__init__(*args, **kwargs)
        self.project = project

    def work(self, burst=False, logging_level="WARN"):
        config = init_config(self.project)
        project = self.project
        redis_host = config[project]["redis_host"]
        redis_port = config[project]["redis_port"]
        v_config = config[project]
        password = os.getenv("REDIS_PASSWD")
        if password is None:
            logger.warning("REDIS_PASSWD environment variable is not set. Using default password.")
            raise Exception("REDIS_PASSWD environment variable is not set. Cannot start worker.")
        batch_size = int(os.getenv("BATCH_SIZE", 32))
        logger.info(f"Connecting to redis at {redis_host}:{redis_port}")
        logger.info(f"Redis queue for project {project} created successfully")
        redis_conn = redis.Redis(host=redis_host, port=redis_port, password=password)
        predictor = ViTWrapper(redis_conn, device=v_config["device"], model_name=v_config["model"], reset=False, batch_size=batch_size)
        _predictor_stack.push(predictor)
        return super().work(burst, logging_level)


def _bytes_to_temp_paths(image_data: List[bytes], filenames: List[str]) -> List[str]:
    """Write image bytes to temp files and return their paths. Caller must unlink paths when done."""
    paths = []
    for data, name in zip(image_data, filenames):
        suffix = Path(name).suffix or ".png"
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            if hasattr(data, "read"):
                data.seek(0)
                raw = data.read()
            else:
                raw = data
            os.write(fd, raw)
        finally:
            os.close(fd)
        paths.append(path)
    return paths


def predict_on_cpu_or_gpu(v_config: dict, image_data: List[bytes], top_n: int, filenames: List[str]) -> dict:
    temp_paths = _bytes_to_temp_paths(image_data, filenames)
    try:
        logger.info(f"Predicting on {len(temp_paths)} images with top_n={top_n} using model {v_config['model']} on device {v_config['device']}")
        predictor = _predictor_stack.top
        predictions, scores, ids = predictor.predict(temp_paths, top_n)
        gc.collect()

        # Use the current date and time (hourly granularity) for output directory and time with nanoseconds for the filename to ensure uniqueness
        current_time_hr = datetime.now().strftime("%Y%m%d_%H0000")
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S") + f".{time.time_ns() % 10**9:09d}"

        # Create output directory if it doesn't exist
        output_dir = v_config.get("output_path", "output")
        output_path = Path(output_dir) / current_time_hr
        output_path.mkdir(parents=True, exist_ok=True)
        output_json = output_path / f"{current_time}.json"

        logger.debug(f"Saving predictions to {current_time}.json")
        with output_json.open("w") as f:
            json.dump({"filenames": filenames, "predictions": predictions, "scores": scores, "ids": ids}, f, indent=4)
        logger.debug(f"Predictions saved to {output_json}")
        output_final = {
            "filenames": filenames,
            "predictions": predictions,
            "scores": scores,
            "ids": ids,
            "output_path": str(output_json),
        }
        return output_final
    except Exception as e:
        error_message = f"Error during prediction: {e}\n{traceback.format_exc()}"
        logger.info(error_message)
        return error_message
    finally:
        for p in temp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


def get_embeddings_task(v_config: dict, image_data: List[bytes], filenames: List[str]) -> dict:
    temp_paths = _bytes_to_temp_paths(image_data, filenames)
    try:
        logger.info(f"Getting embeddings for {len(temp_paths)} images using model {v_config['model']} on device {v_config['device']}")
        predictor = _predictor_stack.top
        embeddings, failed_filenames = predictor.get_embeddings(temp_paths, filenames)
        gc.collect()

        output_final = {
            "filenames": filenames,
            "embeddings": embeddings,
            "failed_filenames": failed_filenames,
        }
        return output_final
    except Exception as e:
        error_message = f"Error during embedding generation: {e}\n{traceback.format_exc()}"
        logger.info(error_message)
        return error_message
    finally:
        for p in temp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
