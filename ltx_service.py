"""
Сборка графа ComfyUI под LTX 2.3 Director по списку сцен.
Сетевая часть (отправка/ожидание/скачивание) — в comfy_client.py.
"""

import json
import copy
import math
import time
import uuid

from comfy_client import COMFY_HOST, submit_job, wait_for_result, upload_image, extract_video_info

FRAME_RATE = 24                        # значение по умолчанию (совпадает с template LTX_API.json)
WORKFLOW_PATH = "LTX_API.json"         # ваш экспортированный воркфлоу
ANCHOR_IMAGE_DURATION_SEC = 1.0        # длительность image-якоря перед сценой с референсом

# Официально LTX 2.3: ширина/высота должны быть кратны 32 (иначе генерация падает —
# см. Lightricks/LTX-2.3 на HuggingFace и модель-кард). Известные "якорные" размеры
# из практики сообщества и официальных ComfyUI-воркфлоу — базовый проход ~960x544,
# после встроенного 2x-апскейлера (LTXVLatentUpsampler) — ~1920x1088. Модель заявлена
# как поддерживающая портрет/ландшафт вплоть до 4K, но чем выше разрешение, тем больше
# VRAM и время рендера — практического потолка для вашего железа мы пока не замеряли.
LTX_ASPECT_RATIOS = {
    "16:9": (16, 9),
    "9:16": (9, 16),
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "21:9": (21, 9),
}
LTX_MIN_MEGAPIXELS = 0.1
LTX_MAX_MEGAPIXELS = 2.0   # консервативный потолок ДО встроенного 2x-апскейла; не проверялось эмпирически выше этого


def resolve_resolution(aspect_ratio: str, megapixels: float) -> tuple[int, int]:
    """
    Считает width/height под заданное соотношение сторон и целевую площадь (в мегапикселях),
    округляя оба значения вверх до кратности 32 (жёсткое требование LTX 2.3).
    """
    if aspect_ratio not in LTX_ASPECT_RATIOS:
        raise ValueError(
            f"Неизвестный aspect_ratio: {aspect_ratio!r}. "
            f"Допустимые значения: {', '.join(LTX_ASPECT_RATIOS)}"
        )
    megapixels = max(LTX_MIN_MEGAPIXELS, min(LTX_MAX_MEGAPIXELS, megapixels))

    rw, rh = LTX_ASPECT_RATIOS[aspect_ratio]
    total_px = megapixels * 1_000_000
    width = math.sqrt(total_px * rw / rh)
    height = math.sqrt(total_px * rh / rw)

    def snap32(x: float) -> int:
        return max(32, round(x / 32) * 32)

    return snap32(width), snap32(height)


def expand_scenes_with_images(scenes: list[dict]) -> list[dict]:
    """
    Если у текстовой сцены есть прикреплённая картинка (scene['image_filename']),
    вставляем перед ней отдельный короткий image-сегмент-якорь. Именно так была
    устроена структура в вашем подтверждённо рабочем экспорте с image-сегментами.
    """
    expanded = []
    for scene in scenes:
        if scene.get("type", "text") == "text" and scene.get("image_filename"):
            expanded.append({
                "type": "image",
                "image_filename": scene["image_filename"],
                "duration_sec": ANCHOR_IMAGE_DURATION_SEC,
            })
        expanded.append(scene)
    return expanded


def seconds_to_frames(seconds: float, frame_rate: int = FRAME_RATE) -> int:
    """
    Переводит секунды в кадры и округляет вверх до кратности 8.
    LTX работает в латентном пространстве блоками по 8 кадров,
    поэтому произвольная длина типа '3.4 сек' может сломать генерацию.
    """
    frames = round(seconds * frame_rate)
    remainder = frames % 8
    if remainder != 0:
        frames += (8 - remainder)
    return frames


def build_timeline(scenes: list[dict], global_prompt: str, frame_rate: int = FRAME_RATE) -> dict:
    """
    Превращает простой список сцен в три поля, которые ждёт узел LTXDirector:
    timeline_data (JSON-строка), local_prompts (строка через ' | '),
    segment_lengths (строка через запятую) + общую длительность.

    scenes — список словарей вида:
      {"type": "text", "prompt": "...", "duration_sec": 3}
      {"type": "image", "image_filename": "имя_уже_загруженного_файла.jpg", "duration_sec": 1}
    """
    segments = []
    local_prompts_parts = []
    segment_lengths_parts = []
    cursor = 0  # текущая позиция в кадрах, растёт по мере добавления сцен

    for scene in scenes:
        length = seconds_to_frames(scene["duration_sec"], frame_rate)

        segment = {
            "id": f"seg_{uuid.uuid4().hex[:12]}",
            "start": cursor,
            "length": length,
            "type": scene["type"],
        }

        if scene["type"] == "text":
            segment["prompt"] = scene["prompt"]
            local_prompts_parts.append(scene["prompt"])

        elif scene["type"] == "image":
            # Картинка должна быть заранее загружена в ComfyUI через /upload/image —
            # см. функцию upload_image() ниже.
            segment["prompt"] = ""
            segment["imageFile"] = scene["image_filename"]
            segment["imageB64"] = f"/api/view?filename={scene['image_filename']}&type=input"
            local_prompts_parts.append("")

        else:
            raise ValueError(f"Неизвестный тип сцены: {scene['type']}")

        segments.append(segment)
        segment_lengths_parts.append(str(length))
        cursor += length

    total_frames = cursor
    total_seconds = total_frames / frame_rate

    timeline_data = {
        "mainTrackEnabled": True,
        "audioTrackEnabled": True,
        "motionTrackEnabled": True,
        "propHeight": 90,
        "globalPropHeight": 60,
        "showFilenames": True,
        "overrideAudio": False,
        "inpaint_audio": True,
        "global_prompt": global_prompt,
        "retake_global_prompt": "",
        "retakeMode": False,
        "retakeStart": 24,
        "retakeLength": 48,
        "retakePrompt": "",
        "retakeStrength": 1,
        "retakeVideo": None,
        "normalStartFrame": 0,
        "normalDurationFrames": total_frames,
        "segments": segments,
        "motionSegments": [],
        "audioSegments": [],
    }

    # guide_strength относится к картиночным сегментам-якорям (сила привязки к опорному
    # кадру), а не к текстовым. Если картинок в таймлайне нет — поле должно быть пустым,
    # иначе (судя по сравнению рабочего и нерабочего экспорта) это может портить результат.
    image_segment_count = sum(1 for s in scenes if s["type"] == "image")

    return {
        "start_second": 0,
        "end_second": total_seconds,
        "duration_seconds": total_seconds,
        "start_frame": 0,
        "end_frame": total_frames,
        "duration_frames": total_frames,
        "frame_rate": frame_rate,
        "timeline_data": json.dumps(timeline_data, ensure_ascii=False),
        "local_prompts": " | ".join(local_prompts_parts),
        "segment_lengths": ",".join(segment_lengths_parts),
        "guide_strength": ",".join(["1.00"] * image_segment_count),
    }


def build_workflow(
    template: dict,
    scenes: list[dict],
    global_prompt: str,
    aspect_ratio: str | None = None,
    megapixels: float | None = None,
    frame_rate: int = FRAME_RATE,
    seed: int | None = None,
) -> tuple[dict, str]:
    """
    Берёт шаблон воркфлоу (то, что вы экспортировали из ComfyUI),
    подставляет в узел 131 наши сцены, даёт задаче seed
    и уникальное имя файла (чтобы результаты разных пользователей не затирали друг друга).

    aspect_ratio/megapixels — ОПЦИОНАЛЬНЫ. Если не заданы, custom_width/custom_height
    остаются как в шаблоне (0/0 — режим "авто", который мы уже проверили вживую).
    Если заданы — разрешение считается через resolve_resolution() и подставляется
    в узел явно. ВНИМАНИЕ: эта ветка (явное разрешение) пока не проверена реальной
    генерацией — проверьте на первом прогоне, что вывод действительно соответствует
    заданному размеру и не ломает пайплайн апскейла.
    """
    workflow = copy.deepcopy(template)  # ВАЖНО: не менять оригинал, а работать с копией

    scenes = expand_scenes_with_images(scenes)
    timeline_fields = build_timeline(scenes, global_prompt, frame_rate=frame_rate)
    workflow["131"]["inputs"].update(timeline_fields)

    if aspect_ratio is not None and megapixels is not None:
        width, height = resolve_resolution(aspect_ratio, megapixels)
        workflow["131"]["inputs"]["custom_width"] = width
        workflow["131"]["inputs"]["custom_height"] = height

    workflow["30"]["inputs"]["noise_seed"] = seed if seed is not None else int(time.time() * 1000) % (2**32)

    job_id = uuid.uuid4().hex[:10]
    workflow["37"]["inputs"]["filename_prefix"] = f"video/job_{job_id}"

    return workflow, job_id


def result_info(outputs: dict) -> dict:
    """filename/subfolder/type готового видео. Узел SaveVideo в LTX-воркфлоу — id '37'."""
    return extract_video_info(outputs, save_video_node_id="37")


if __name__ == "__main__":
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        template = json.load(f)

    scenes = [
        {
            "type": "text",
            "prompt": "детектив в плаще стоит на крыше небоскрёба ночью, вид сверху на город в дожде",
            "duration_sec": 4,
        },
        {
            "type": "text",
            "prompt": "средний план: детектив достаёт сигарету, неоновая вывеска мигает позади",
            "duration_sec": 2.5,
        },
        {
            "type": "text",
            "prompt": "крупный план глаз детектива, отражение полицейских мигалок",
            "duration_sec": 3,
        },
    ]

    workflow, job_id = build_workflow(
        template,
        scenes,
        global_prompt="Кинематографичный неоновый нуар, дождь, реалистичное освещение",
    )

    prompt_id = submit_job(workflow)
    print(f"Задача отправлена. job_id={job_id}, prompt_id={prompt_id}")

    outputs = wait_for_result(prompt_id)
    info = result_info(outputs)
    print("Готово! Файл на ComfyUI:")
    print(f"{COMFY_HOST}/view?filename={info['filename']}&subfolder={info['subfolder']}&type={info['type']}")
