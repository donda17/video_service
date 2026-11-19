"""
Веб-сервер поверх ltx_service.py и minimax_service.py.

Запуск:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Проверка в браузере после запуска:
    http://127.0.0.1:8000/docs
"""

import json
import os
import queue
import threading
import time
import urllib.parse
import uuid

from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import comfy_client
from comfy_client import submit_job, wait_for_result, upload_image
import ltx_service
import minimax_service

# --- Настройки ---

MAX_TOTAL_DURATION_SEC = 60   # лимит для LTX (до минуты)
MAX_SCENES = 12               # потолок, чтобы не собирать гигантский граф

API_KEY = os.environ.get("LTX_API_KEY")


def check_api_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Неверный или отсутствующий API-ключ")


with open(ltx_service.WORKFLOW_PATH, "r", encoding="utf-8") as f:
    LTX_TEMPLATE = json.load(f)

with open(minimax_service.WORKFLOW_PATH, "r", encoding="utf-8") as f:
    MINIMAX_TEMPLATE = json.load(f)


# --- Схемы запросов ---

class SceneIn(BaseModel):
    type: str = Field(description="'text' или 'image'")
    prompt: str = ""
    duration_sec: float
    image_filename: str | None = None


class LtxGenerateRequest(BaseModel):
    scenes: list[SceneIn]
    global_prompt: str = ""
    aspect_ratio: str | None = None   # None = авто 
    megapixels: float | None = None   # оба должны быть заданы вместе, иначе игнорируются


class MinimaxGenerateRequest(BaseModel):
    prompt: str
    duration_sec: float = 8
    aspect_ratio: str = "16:9 (Widescreen)"
    megapixels: float = minimax_service.MINIMAX_DEFAULT_MEGAPIXELS


# --- "База данных" задач + очередь на одну GPU ---
# Общая для всех моделей: если одновременно попросят LTX и MiniMax,
# они всё равно не должны выполняться параллельно на одной видеокарте.

jobs: dict[str, dict] = {}
job_queue: "queue.Queue[tuple[str, callable]]" = queue.Queue()


def worker_loop() -> None:
    while True:
        job_id, generate_fn = job_queue.get()
        started_at = time.time()
        jobs[job_id]["started_at"] = started_at
        jobs[job_id]["queue_wait_sec"] = round(started_at - jobs[job_id]["queued_at"], 1)
        try:
            jobs[job_id]["status"] = "processing"
            video_url = generate_fn()
            jobs[job_id]["status"] = "done"
            jobs[job_id]["video_url"] = video_url
        except Exception as e:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
        finally:
            finished_at = time.time()
            jobs[job_id]["finished_at"] = finished_at
            jobs[job_id]["generation_sec"] = round(finished_at - started_at, 1)
        job_queue.task_done()


threading.Thread(target=worker_loop, daemon=True).start()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return FileResponse("index.html")


@app.post("/upload-image", dependencies=[Depends(check_api_key)])
def upload_image_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Ожидается файл изображения")

    file_bytes = file.file.read()
    max_bytes = 10 * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(400, "Файл слишком большой (максимум 10 МБ)")

    stored_filename = upload_image(file_bytes, file.filename)
    return {"filename": stored_filename}


def video_proxy_url(info: dict) -> str:
    query = urllib.parse.urlencode(info)
    return f"/video?{query}"


@app.get("/video")
def get_video(filename: str, subfolder: str = "", type: str = "output"):
    try:
        upstream = comfy_client.fetch_video(filename, subfolder, type)
    except Exception as e:
        raise HTTPException(502, f"Не удалось получить видео с ComfyUI: {e}")
    return StreamingResponse(upstream.iter_content(chunk_size=65536), media_type="video/mp4")


@app.post("/generate", dependencies=[Depends(check_api_key)])
def generate_ltx(request: LtxGenerateRequest):
    anchors_count = sum(1 for s in request.scenes if s.image_filename)
    total_duration = (
        sum(s.duration_sec for s in request.scenes)
        + anchors_count * ltx_service.ANCHOR_IMAGE_DURATION_SEC
    )

    if not request.scenes:
        raise HTTPException(400, "Список сцен пуст")
    if len(request.scenes) > MAX_SCENES:
        raise HTTPException(400, f"Слишком много сцен (максимум {MAX_SCENES})")
    if total_duration > MAX_TOTAL_DURATION_SEC:
        raise HTTPException(
            400,
            f"Суммарная длительность {total_duration} сек превышает лимит {MAX_TOTAL_DURATION_SEC} сек",
        )
    if request.aspect_ratio is not None and request.aspect_ratio not in ltx_service.LTX_ASPECT_RATIOS:
        raise HTTPException(
            400,
            f"Неизвестный aspect_ratio. Допустимые значения: {', '.join(ltx_service.LTX_ASPECT_RATIOS)}",
        )

    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "video_url": None, "error": None, "queued_at": time.time(), "model": "ltx-director"}

    scenes_as_dicts = [s.model_dump() for s in request.scenes]

    def do_generate():
        workflow, _ = ltx_service.build_workflow(
            LTX_TEMPLATE,
            scenes_as_dicts,
            request.global_prompt,
            aspect_ratio=request.aspect_ratio,
            megapixels=request.megapixels,
        )
        prompt_id = submit_job(workflow)
        outputs = wait_for_result(prompt_id)
        return video_proxy_url(ltx_service.result_info(outputs))

    job_queue.put((job_id, do_generate))
    return {"job_id": job_id, "status": "queued", "model": "ltx-director"}


@app.post("/generate-minimax", dependencies=[Depends(check_api_key)])
def generate_minimax(request: MinimaxGenerateRequest):
    if not request.prompt.strip():
        raise HTTPException(400, "Промпт не может быть пустым")
    if not (minimax_service.MIN_DURATION_SEC <= request.duration_sec <= minimax_service.MAX_DURATION_SEC):
        raise HTTPException(
            400,
            f"Длительность должна быть от {minimax_service.MIN_DURATION_SEC} "
            f"до {minimax_service.MAX_DURATION_SEC} сек — это ограничение самой модели",
        )
    if request.aspect_ratio not in minimax_service.MINIMAX_ASPECT_RATIOS:
        raise HTTPException(
            400,
            f"Неизвестный aspect_ratio. Допустимые значения: {', '.join(minimax_service.MINIMAX_ASPECT_RATIOS)}",
        )

    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "video_url": None, "error": None, "queued_at": time.time(), "model": "minimax-h3"}

    def do_generate():
        workflow, _ = minimax_service.build_workflow(
            MINIMAX_TEMPLATE,
            request.prompt,
            request.duration_sec,
            aspect_ratio=request.aspect_ratio,
            megapixels=request.megapixels,
        )
        prompt_id = submit_job(workflow)
        outputs = wait_for_result(prompt_id)
        return video_proxy_url(minimax_service.result_info(outputs))

    job_queue.put((job_id, do_generate))
    return {"job_id": job_id, "status": "queued", "model": "minimax-h3"}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Задача с таким job_id не найдена")
    return job


@app.get("/metrics")
def get_metrics():
    summary = {"total": len(jobs), "by_status": {}, "by_model": {}}
    for job in jobs.values():
        summary["by_status"][job["status"]] = summary["by_status"].get(job["status"], 0) + 1
        model = job.get("model") or "unknown"
        bucket = summary["by_model"].setdefault(model, {"count": 0, "done": 0, "error": 0, "generation_sec": []})
        bucket["count"] += 1
        if job["status"] == "done":
            bucket["done"] += 1
            if "generation_sec" in job:
                bucket["generation_sec"].append(job["generation_sec"])
        elif job["status"] == "error":
            bucket["error"] += 1

    for model, bucket in summary["by_model"].items():
        times = bucket.pop("generation_sec")
        if times:
            bucket["avg_generation_sec"] = round(sum(times) / len(times), 1)
            bucket["min_generation_sec"] = round(min(times), 1)
            bucket["max_generation_sec"] = round(max(times), 1)

    return {"summary": summary, "jobs": jobs}
