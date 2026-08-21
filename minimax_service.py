"""
Сборка графа ComfyUI под MiniMax H3 (базовый T2V, без референс-картинок и без
стороннего ускорителя Spectrum — см. обсуждение "голой" модели).

В отличие от LTX Director, тут нет списка сцен — один промпт на весь ролик,
максимум 15 секунд (это жёсткое ограничение самой модели, не наше).
"""

import copy
import json
import time
import uuid

from comfy_client import COMFY_HOST, submit_job, wait_for_result, upload_image, extract_video_info

WORKFLOW_PATH = "Minimax_h3_t2v_i2v_api.json"
MAX_DURATION_SEC = 15   # жёсткий потолок самой модели, не наш лимит
MIN_DURATION_SEC = 1    # снижено по запросу; формула узла 132 сама не даёт уйти ниже ~5 кадров,
                         # так что и 1 сек технически безопасен — просто вне официально
                         # заявленного тренировочного диапазона (4-15 сек), результат может быть
                         # менее предсказуем, чем внутри этого диапазона

# Точный список допустимых значений узла ResolutionSelector (встроенный узел ComfyUI,
# см. docs.comfy.org/built-in-nodes/ResolutionSelector) — ComfyUI строго валидирует
# combo-поля и отклоняет всё, чего нет в этом списке буква в букву.
MINIMAX_ASPECT_RATIOS = [
    "1:1 (Square)",
    "2:3 (Portrait Photo)",
    "3:2 (Photo)",
    "3:4 (Portrait Standard)",
    "4:3 (Standard)",
    "9:16 (Portrait Widescreen)",
    "16:9 (Widescreen)",
    "21:9 (Ultrawide)",
]
# Официальный локальный релиз (H3-Base) заявлен как "768p-class": короткая сторона
# ограничена ~768px (до 768x1344), полный 2K-пайплайн (Regenerate-2K) в открытых
# весах пока не выпущен. Поэтому потолок megapixels держим консервативным —
# 768x1344 ≈ 1.03 MP — а не тот широкий диапазон (0.1-16.0), что формально
# разрешает сам узел ResolutionSelector.
MINIMAX_MIN_MEGAPIXELS = 0.1
MINIMAX_MAX_MEGAPIXELS = 1.05
MINIMAX_DEFAULT_MEGAPIXELS = 0.4   # официальная рекомендация: стартовать с превью 0.4 MP


def build_workflow(
    template: dict,
    prompt: str,
    duration_sec: float,
    aspect_ratio: str = "16:9 (Widescreen)",
    megapixels: float = MINIMAX_DEFAULT_MEGAPIXELS,
    seed: int | None = None,
) -> tuple[dict, str]:
    """
    duration_sec можно передавать как есть — округление до нужной "сетки"
    кадров (блоки по 17 + 5, см. узел 132) считает сам ComfyUI внутри графа,
    нам достаточно просто передать секунды в узел 133 (PrimitiveFloat).
    """
    if aspect_ratio not in MINIMAX_ASPECT_RATIOS:
        raise ValueError(
            f"Неизвестный aspect_ratio: {aspect_ratio!r}. "
            f"Допустимые значения: {', '.join(MINIMAX_ASPECT_RATIOS)}"
        )

    duration_sec = max(MIN_DURATION_SEC, min(MAX_DURATION_SEC, duration_sec))
    megapixels = max(MINIMAX_MIN_MEGAPIXELS, min(MINIMAX_MAX_MEGAPIXELS, megapixels))

    workflow = copy.deepcopy(template)

    workflow["131"]["inputs"]["prompt"] = prompt
    workflow["133"]["inputs"]["value"] = duration_sec
    workflow["115"]["inputs"]["aspect_ratio"] = aspect_ratio
    workflow["115"]["inputs"]["megapixels"] = megapixels
    workflow["129"]["inputs"]["noise_seed"] = seed if seed is not None else int(time.time() * 1000) % (2**32)

    job_id = uuid.uuid4().hex[:10]
    workflow["92"]["inputs"]["filename_prefix"] = f"video/job_{job_id}"

    return workflow, job_id


def result_info(outputs: dict) -> dict:
    """filename/subfolder/type готового видео. Узел SaveVideo в MiniMax-воркфлоу — id '92'."""
    return extract_video_info(outputs, save_video_node_id="92")


if __name__ == "__main__":
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        template = json.load(f)

    workflow, job_id = build_workflow(
        template,
        prompt=(
            "детектив в плаще стоит на крыше ночного города под дождём, "
            "неоновые вывески отражаются в лужах, кинематографично, "
            "звук: далёкие сирены, шум дождя"
        ),
        duration_sec=8,
    )

    prompt_id = submit_job(workflow)
    print(f"Задача отправлена. job_id={job_id}, prompt_id={prompt_id}")

    outputs = wait_for_result(prompt_id)
    info = result_info(outputs)
    print("Готово! Файл на ComfyUI:")
    print(f"{COMFY_HOST}/view?filename={info['filename']}&subfolder={info['subfolder']}&type={info['type']}")
