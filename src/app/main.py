# fastapi-vss, Apache-2.0 license
# Filename: app/main.py
# Description: Process images with Vision Transformer (ViT) models
import asyncio
import json
import logging
import os
import time
import warnings

import redis
import torch
from fastapi import FastAPI, status, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from prometheus_fastapi_instrumentator import Instrumentator
from rq import Queue
from rq.job import Job

from app import __version__
from app import logger
from app.config import init_config, BATCH_SIZE
from app.logger import info, debug
from app.predictors.tasks import predict_on_cpu_or_gpu, get_embeddings_task
from app.predictors.vector_similarity import VectorSimilarity

# Suppress the FutureWarning from torch about pynvml - we're using nvidia-ml-py
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*")

try:
    import pynvml
except ImportError:
    pynvml = None

log_path = os.getenv("LOG_DIR", "logs")
logger = logger.create_logger_file(log_path)

# Suppress verbose multipart parsing debug logs from python_multipart
logging.getLogger("python_multipart.multipart").setLevel(logging.WARNING)
# Suppress verbose PIL/Pillow PNG stream debug logs
logging.getLogger("PIL").setLevel(logging.WARNING)

info(f"Starting Fast-VSS API version {__version__}")

# Origins allowed for CORS (e.g. when behind a reverse proxy). Comma-separated list via FASTAPI_VSS_CORS_ORIGINS.
CORS_ORIGINS: List[str] = []
_cors_env = os.environ.get("FASTAPI_VSS_CORS_ORIGINS", "").strip()
if _cors_env:
    CORS_ORIGINS.extend(origin.strip() for origin in _cors_env.split(",") if origin.strip())

app = FastAPI(
    title=f"Fast-VSS API version {__version__}",
    description=f"""Run vector similarity search using Vision Transformer (ViT) models . Version {__version__}""",
    version=__version__,
)
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

Instrumentator().instrument(app).expose(app)

info("Loading configuration")
config = init_config()

if len(config) == 0:
    raise Exception("No projects found in the configuration file")

queues = {}
connections = {}

# WebSocket poll interval in seconds - short enough to be responsive, avoids hammering Redis.
# This doubles as the heartbeat interval: a client cannot distinguish a slow job from a dead
# connection by the job's duration, only by the gap between frames, so it must stay well below
# whatever idle timeout clients use (fiftyone-sync: FASTVSS_WS_IDLE_TIMEOUT, default 120s).
WS_POLL_INTERVAL = float(os.getenv("WS_POLL_INTERVAL", "0.5"))
# Maximum time to wait for a job to complete before closing the WebSocket.
#
# Jobs are processed by a single serial RQ worker per project (see start_worker.py), so a job
# submitted while several others are queued ahead of it legitimately waits for all of them.
# The old 300s was well under what a backed-up queue needs and cut clients off mid-job. Keep
# this at or above the client's own per-job budget, or the client's budget is meaningless.
WS_MAX_WAIT = float(os.getenv("WS_MAX_WAIT", "1800"))

for project in config.keys():
    redis_host = config[project]["redis_host"]
    redis_port = config[project]["redis_port"]
    device = config[project]["device"]
    password = os.getenv("REDIS_PASSWD")
    info(f"Connecting to redis at {redis_host}:{redis_port}")
    redis_conn = redis.Redis(host=redis_host, port=redis_port, password=password)
    connections[project] = redis_conn
    info(f"Creating Redis queue for project {project}")
    redis_queue = Queue(connection=redis_conn)
    info(f"Redis queue for project {project} created successfully")
    queues[project] = redis_queue

DEFAULT_PROJECT = list(config.keys())[0]

GPU_AVAILABLE = False
if torch.cuda.is_available() and pynvml is not None:
    pynvml.nvmlInit()
    GPU_AVAILABLE = True


@app.get("/")
async def root():
    return {"message": f"Welcome to Fast-VSS API version {__version__}"}


@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    """
    Health check endpoint to verify if the API is running
    """
    return {"status": "ok", "version": __version__}


@app.get("/gpu-memory")
def gpu_memory():
    if not GPU_AVAILABLE:
        return {"error": "No GPU available"}
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {"used_memory": mem_info.used, "total_memory": mem_info.total}


@app.get("/projects")
async def get_projects():
    return {"projects": list(config.keys())}


@app.get("/ids/{project}", status_code=status.HTTP_200_OK)
async def get_ids(project: str = DEFAULT_PROJECT):
    """
    Get the first 100 IDs and their classes for a given project.
    This endpoint returns a limited sample of IDs for quick inspection.
    For complete data exports, please use the appropriate bulk export tools.
    """
    # Check if the project name is in the config
    if project not in config.keys():
        return {"error": f"Invalid project name {project}"}

    try:
        # Connect to the Redis queue for the project
        redis_conn = connections[project]
        info(f"Fetching IDs for project {project}")
        all_keys = redis_conn.keys(f"{VectorSimilarity.DOC_PREFIX}*")
        # Data is formatted <doc:label:id>, e.g. doc:Otter:12467, doc:Otter:12467, etc.
        classes = []
        ids = []
        for i, key in enumerate(all_keys):
            str = key.decode("utf-8").split(":")
            if len(str) == 3:
                classes.append(str[1])
                ids.append(str[2])
            # Limit to first 100 results
            if len(ids) >= 100:
                break

        return {"ids": ids, "classes": classes}
    except Exception as e:
        return {"error": f"Error getting ids: {e}"}


@app.post("/knn/{top_n}/{project}", status_code=status.HTTP_200_OK)
async def knn(files: List[UploadFile] = File(...), top_n: int = 1, project: str = DEFAULT_PROJECT):
    try:
        # Check if the project name is in the config
        if project not in config.keys():
            return {"error": f"Invalid project name {project}"}

        info(f"Predicting {len(files)} for top {top_n} in project {project}")
        if len(files) > BATCH_SIZE:
            return {"error": f"Images should be less than batch size {BATCH_SIZE}"}

        if top_n == 0:
            return {"error": "Please provide a valid top_n value greater than 0"}

        image_bytes = [await f.read() for f in files]
        filenames = [f.filename for f in files]
        redis_queue = queues[project]

        info(f"Enqueuing job for {len(image_bytes)} images with top_n={top_n} in project {project}")
        vss_config = config[project]
        job = redis_queue.enqueue(predict_on_cpu_or_gpu, vss_config, image_bytes, top_n, filenames)
        job_id = job.id
        debug(f"Enqueued job with ID {job_id} for project {project}")
        return {"job_id": job_id, "Comment": f"Use /predict/job/{job_id}/{project} to check status."}
    except Exception as e:
        return {"error": f"Error predicting images: {e}"}


@app.post("/embed/{project}", status_code=status.HTTP_200_OK)
async def embeddings(files: List[UploadFile] = File(...), project: str = DEFAULT_PROJECT):
    try:
        # Check if the project name is in the config
        if project not in config.keys():
            return {"error": f"Invalid project name {project}"}

        info(f"Getting embeddings for {len(files)} in project {project}")
        if len(files) > BATCH_SIZE:
            return {"error": f"Images should be less than batch size {BATCH_SIZE}"}

        image_bytes = [await f.read() for f in files]
        filenames = [f.filename for f in files]
        redis_queue = queues[project]

        info(f"Enqueuing embedding job for {len(image_bytes)} images in project {project}")
        vss_config = config[project]
        job = redis_queue.enqueue(get_embeddings_task, vss_config, image_bytes, filenames)
        job_id = job.id
        debug(f"Enqueued embedding job with ID {job_id} for project {project}")
        return {"job_id": job_id, "Comment": f"Use /predict/job/{job_id}/{project} to check status."}
    except Exception as e:
        return {"error": f"Error getting embeddings: {e}"}


@app.get("/predict/job/{job_id}/{project}")
async def get_job_result(job_id: str, project: str = DEFAULT_PROJECT):
    if project not in config.keys():
        return {"error": f"Invalid project name {project}"}

    try:
        # Check if the job ID is valid
        if not await asyncio.to_thread(
            Job.exists, job_id, connection=connections[project]
        ):
            return {"error": f"Job ID {job_id} does not exist in project {project}"}

        redis_conn = connections[project]
        info(f"Fetching job status for job ID {job_id} in project {project}")
        job = await asyncio.to_thread(Job.fetch, job_id, connection=redis_conn)
        if job.is_finished:
            return {
                "status": "done",
                "result": await asyncio.to_thread(job.return_value),
            }
        elif job.is_failed:
            return {"status": "failed"}
        else:
            return {"status": "pending"}
    except Exception as e:
        return {"error": f"Error fetching job status: {e}"}


@app.websocket("/ws/predict/job/{job_id}/{project}")
async def ws_job_result(websocket: WebSocket, job_id: str, project: str = DEFAULT_PROJECT):
    """
    WebSocket endpoint to stream job status updates.
    Sends JSON messages: {"status": "pending"}, {"status": "done", "result": ...},
    or {"status": "failed"} / {"status": "error", "message": ...}
    Closes the connection once the job reaches a terminal state.
    """
    await websocket.accept()

    if project not in config.keys():
        await websocket.send_text(json.dumps({"status": "error", "message": f"Invalid project name {project}"}))
        await websocket.close()
        return

    redis_conn = connections[project]

    # Job.exists / Job.fetch / job.return_value() are blocking redis-py calls, and this is an
    # async endpoint sharing one event loop with every other request this process is serving.
    # Running them inline stalls all of those -- including the heartbeat frames other clients
    # rely on to tell a slow job from a dead connection -- so they go to a worker thread.
    # redis-py clients are thread-safe (they hold a connection pool), as is unpickling a
    # result, which for a batch of embeddings is itself far from free.
    if not await asyncio.to_thread(Job.exists, job_id, connection=redis_conn):
        await websocket.send_text(json.dumps({"status": "error", "message": f"Job ID {job_id} does not exist in project {project}"}))
        await websocket.close()
        return

    # Wall-clock, not a count of loop iterations. Incrementing a counter by WS_POLL_INTERVAL
    # (as this used to) measures only the sleeps and ignores how long each Redis round-trip
    # took, so the effective limit drifted arbitrarily far past WS_MAX_WAIT under load --
    # exactly when an accurate limit matters.
    start = time.monotonic()
    try:
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= WS_MAX_WAIT:
                info(f"WebSocket job {job_id} timed out in project {project} after {elapsed:.0f}s")
                await websocket.send_text(json.dumps({
                    "status": "error",
                    "message": f"Timed out waiting for job after {elapsed:.0f}s",
                }))
                break

            job = await asyncio.to_thread(Job.fetch, job_id, connection=redis_conn)

            if job.is_finished:
                info(f"WebSocket job {job_id} finished in project {project} after {elapsed:.0f}s")
                result = await asyncio.to_thread(job.return_value)
                await websocket.send_text(json.dumps({"status": "done", "result": result}))
                break
            elif job.is_failed:
                info(f"WebSocket job {job_id} failed in project {project} after {elapsed:.0f}s")
                await websocket.send_text(json.dumps({
                    "status": "failed",
                    "message": f"Job {job_id} failed in project {project}",
                }))
                break
            else:
                # Heartbeat. Clients time out on the gap between frames, so this must keep
                # flowing for the whole wait, however long the job itself takes.
                await websocket.send_text(json.dumps({"status": "pending"}))

            await asyncio.sleep(WS_POLL_INTERVAL)
    except WebSocketDisconnect:
        debug(f"WebSocket client disconnected for job {job_id} in project {project}")
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"status": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
