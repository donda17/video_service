"""
Общая часть, одинаковая для любой модели: отправка графа в ComfyUI,
ожидание результата, скачивание готового видео, загрузка референс-картинок.
Конкретные модели (ltx_service.py, minimax_service.py) импортируют это отсюда,
а сами отвечают только за то, как собрать граф под свою модель.
"""

import time
import requests

COMFY_HOST = "http://127.0.0.1:8188"   # адрес машины с ComfyUI
COMFY_HOST = COMFY_HOST.rstrip("/")


def submit_job(workflow: dict) -> str:
    """Отправляет граф в ComfyUI. Возвращает prompt_id — идентификатор задачи в очереди."""
    resp = requests.post(f"{COMFY_HOST}/prompt", json={"prompt": workflow})
    if not resp.ok:
        # ComfyUI обычно возвращает подробное описание проблемы (какой узел, какое
        # поле, какое значение не подошло) прямо в теле ответа — requests.raise_for_status()
        # это тело выбрасывает и оставляет только код статуса, что бесполезно для отладки.
        try:
            details = resp.json()
        except ValueError:
            details = resp.text
        raise RuntimeError(f"ComfyUI отклонил граф ({resp.status_code}): {details}")
    return resp.json()["prompt_id"]


def wait_for_result(prompt_id: str, poll_interval: float = 3.0, timeout: float = 5400.0) -> dict:
    """Раз в poll_interval секунд спрашивает у ComfyUI: "готово?", возвращает outputs, когда да."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"{COMFY_HOST}/history/{prompt_id}")
        resp.raise_for_status()
        history = resp.json()
        if prompt_id in history:
            return history[prompt_id]["outputs"]
        time.sleep(poll_interval)
    raise TimeoutError("Генерация не завершилась за отведённое время")


def extract_video_info(outputs: dict, save_video_node_id: str) -> dict:
    """Достаёт filename/subfolder/type готового видео из outputs — без построения URL."""
    video_info = outputs[save_video_node_id]["images"][0]
    return {
        "filename": video_info["filename"],
        "subfolder": video_info.get("subfolder", ""),
        "type": video_info.get("type", "output"),
    }


def fetch_video(filename: str, subfolder: str = "", type_: str = "output"):
    """
    Стримит видео с ComfyUI (requests-ответ с stream=True — вызывающий код сам
    итерируется по .iter_content()). Используется в app.py, чтобы браузер
    забирал видео у НАШЕГО сервиса, а не напрямую у ComfyUI (см. комментарий
    в app.py про SSH-туннели и Cloudflare Tunnel).
    """
    resp = requests.get(
        f"{COMFY_HOST}/view",
        params={"filename": filename, "subfolder": subfolder, "type": type_},
        stream=True,
    )
    resp.raise_for_status()
    return resp


def upload_image(file_bytes: bytes, filename: str) -> str:
    """Загружает картинку в input-папку ComfyUI через его собственный API."""
    resp = requests.post(
        f"{COMFY_HOST}/upload/image",
        files={"image": (filename, file_bytes)},
    )
    resp.raise_for_status()
    return resp.json()["name"]
