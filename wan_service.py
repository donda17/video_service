"""
Сборка графа ComfyUI под Wan 2.2 I2V + SVI Pro (Stable Video Infinity Pro).

В отличие от LTX/MiniMax, это I2V-модель — обязательно нужна референс-картинка
(первый кадр). Каждый следующий сегмент кондиционируется на ЛАТЕНТЕ (не пикселях)
предыдущего сегмента + держит константный якорь на исходное фото весь ролик.
Внутри сегмента — двухфазный семплинг (HIGH-noise модель → LOW-noise модель,
MoE-архитектура Wan 2.2).

ВАЖНО: этот адаптер построен по образцу ОДНОГО присланного экспорта (4 сегмента).
Формат вывода VHS_VideoCombine в /history ЕЩЁ НЕ ПРОВЕРЕН вживую (в отличие от
SaveVideo у LTX/MiniMax, который мы подтвердили эмпirически) — велика вероятность,
что видео лежит под ключом "gifs", а не "images" (известная особенность VHS-нод).
Первый реальный прогон стоит проверить и поправить result_info() при необходимости.
"""

import copy
import json
import time
import uuid

from comfy_client import submit_job, wait_for_result, upload_image

WORKFLOW_PATH = "SVI-Pro_API.json"

FRAME_RATE = 16          # узел 630 в шаблоне — не 24, как у LTX/MiniMax
FRAMES_PER_SEGMENT = 81  # узел 608 — длина одного сегмента в кадрах (~5 сек при 16 fps)
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 480
MAX_SEGMENTS = 8          # мягкий потолок сервиса — каждый сегмент тяжёлый (2 UNet x 2 LoRA)

DEFAULT_NEGATIVE_PROMPT = (
    "Bright, oversaturated tones; overexposed; static; unclear or blurry details; "
    "subtitles; style or artistic work; painting or image; still/frozen frame; "
    "overall grayish appearance; worst quality; low quality; JPEG artifacts; ugly; "
    "incomplete or damaged; extra fingers; poorly drawn hands; poorly drawn face; "
    "deformed; disfigured; malformed limbs; fused fingers; motionless/still image; "
    "messy/cluttered background; three legs; many people in the background; "
    "walking backwards."
)

# Узлы, которые переиспользуются как есть из шаблона — общие для всех сегментов
# (модели, LoRA-стек, референс-картинка, общие параметры семплера).
SHARED_NODE_IDS = [
    "136", "294", "295",              # resize + LoadImage + VAEEncode референс-картинки
    "606", "607", "608", "609", "610", "630",  # width/height/num_frames/steps/split_step/fps
    "662", "663",                     # UnetLoaderGGUF (high/low)
    "664", "665", "666", "667",       # PatchSageAttention + ModelPatchTorchSettings
    "668", "669", "670", "671",       # LoRA-стек (HIGH/LOW x 2 LoRA каждый)
    "672", "673",                     # CLIPLoader, VAELoader
    "623:299", "623:297", "623:298", "623:128",  # KSamplerSelect, ModelSamplingSD3, BasicScheduler, SplitSigmas
]


def build_workflow(
    template: dict,
    prompts: list[str],
    reference_image_filename: str,
    negative_prompt: str | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
) -> tuple[dict, str]:
    """
    prompts — список текстов, один на сегмент (аналог сцен у LTX, но здесь каждый
    сегмент физически кондиционируется на предыдущем через латент, а не через
    отдельный монтажный переход).
    reference_image_filename — имя файла, уже загруженного в ComfyUI через upload_image()
    (это I2V-модель, без стартового изображения не работает).
    """
    if not prompts:
        raise ValueError("Нужен хотя бы один промпт (один сегмент)")
    if len(prompts) > MAX_SEGMENTS:
        raise ValueError(f"Слишком много сегментов (максимум {MAX_SEGMENTS}) — каждый тяжёлый по VRAM/времени")

    negative_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT
    width = width or DEFAULT_WIDTH
    height = height or DEFAULT_HEIGHT
    base_seed = seed if seed is not None else int(time.time() * 1000) % (2**32)

    workflow: dict = {}

    # 1. Копируем общие узлы из шаблона как есть
    for node_id in SHARED_NODE_IDS:
        workflow[node_id] = copy.deepcopy(template[node_id])

    # 2. Референс-картинка и её целевое разрешение — подставляем пользовательские значения
    workflow["294"]["inputs"]["image"] = reference_image_filename
    workflow["606"]["inputs"]["value"] = width
    workflow["607"]["inputs"]["value"] = height

    # 3. Общий негативный промпт — один узел на все сегменты (в оригинале он почти
    # не менялся между сегментами, поэтому не дублируем)
    workflow["neg_shared"] = {
        "inputs": {"text": negative_prompt, "clip": ["672", 0]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Negative, shared)"},
    }

    # 4. Один DisableNoise на все сегменты (узел без состояния, безопасно шарить)
    workflow["disable_noise_shared"] = {
        "inputs": {},
        "class_type": "DisableNoise",
        "_meta": {"title": "DisableNoise"},
    }

    prev_low_sampler_id = None   # латент предыдущего сегмента -> prev_samples следующего
    prev_overlap_source = None   # (node_id, index) — накопленное видео для склейки
    last_vaedecode_id = None

    for i, prompt_text in enumerate(prompts):
        p = f"seg{i}_"

        workflow[p + "pos"] = {
            "inputs": {"text": prompt_text, "clip": ["672", 0]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": f"Positive prompt (segment {i})"},
        }

        workflow[p + "noise"] = {
            "inputs": {"noise_seed": base_seed + i},
            "class_type": "RandomNoise",
            "_meta": {"title": f"RandomNoise (segment {i})"},
        }

        svi_inputs = {
            "length": ["608", 0],
            "motion_latent_count": 1,
            "positive": [p + "pos", 0],
            "negative": ["neg_shared", 0],
            "anchor_samples": ["295", 0],
        }
        if prev_low_sampler_id is not None:
            svi_inputs["prev_samples"] = [prev_low_sampler_id, 0]

        workflow[p + "svi"] = {
            "inputs": svi_inputs,
            "class_type": "WanImageToVideoSVIPro",
            "_meta": {"title": f"WanImageToVideoSVIPro (segment {i})"},
        }

        # HIGH-noise фаза (первая половина шагов)
        workflow[p + "guider_high"] = {
            "inputs": {
                "cfg": 1, "start_percent": 0, "end_percent": 1,
                "model": ["670", 0],
                "positive": [p + "svi", 0], "negative": [p + "svi", 1],
            },
            "class_type": "ScheduledCFGGuidance",
            "_meta": {"title": f"Guidance HIGH (segment {i})"},
        }
        workflow[p + "sampler_high"] = {
            "inputs": {
                "noise": [p + "noise", 0],
                "guider": [p + "guider_high", 0],
                "sampler": ["623:299", 0],
                "sigmas": ["623:128", 0],
                "latent_image": [p + "svi", 2],
            },
            "class_type": "SamplerCustomAdvanced",
            "_meta": {"title": f"Sampler HIGH (segment {i})"},
        }

        # LOW-noise фаза (вторая половина шагов) — её результат идёт дальше по цепочке
        workflow[p + "guider_low"] = {
            "inputs": {
                "cfg": 1, "start_percent": 0, "end_percent": 1,
                "model": ["671", 0],
                "positive": [p + "svi", 0], "negative": [p + "svi", 1],
            },
            "class_type": "ScheduledCFGGuidance",
            "_meta": {"title": f"Guidance LOW (segment {i})"},
        }
        workflow[p + "sampler_low"] = {
            "inputs": {
                "noise": ["disable_noise_shared", 0],
                "guider": [p + "guider_low", 0],
                "sampler": ["623:299", 0],
                "sigmas": ["623:128", 1],
                "latent_image": [p + "sampler_high", 0],
            },
            "class_type": "SamplerCustomAdvanced",
            "_meta": {"title": f"Sampler LOW (segment {i})"},
        }

        workflow[p + "decode"] = {
            "inputs": {"samples": [p + "sampler_low", 0], "vae": ["673", 0]},
            "class_type": "VAEDecode",
            "_meta": {"title": f"VAE Decode (segment {i})"},
        }

        if i == 0:
            prev_overlap_source = (p + "decode", 0)
        else:
            workflow[p + "overlap"] = {
                "inputs": {
                    "overlap": 5,
                    "overlap_side": "source",
                    "overlap_mode": "linear_blend",
                    "source_images": list(prev_overlap_source),
                    "new_images": [p + "decode", 0],
                },
                "class_type": "ImageBatchExtendWithOverlap",
                "_meta": {"title": f"Overlap blend (segment {i})"},
            }
            prev_overlap_source = (p + "overlap", 2)

        prev_low_sampler_id = p + "sampler_low"
        last_vaedecode_id = p + "decode"

    # 5. Финальный узел сохранения — только один, на накопленном/финальном видео
    job_id = uuid.uuid4().hex[:10]
    final_source = list(prev_overlap_source) if len(prompts) > 1 else [last_vaedecode_id, 0]
    workflow["save_video"] = {
        "inputs": {
            "frame_rate": ["630", 0],
            "loop_count": 0,
            "filename_prefix": f"video/job_{job_id}",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": True,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
            "images": final_source,
        },
        "class_type": "VHS_VideoCombine",
        "_meta": {"title": "Save Video (final)"},
    }

    return workflow, job_id


def result_info(outputs: dict) -> dict:
    """
    НЕ ПРОВЕРЕНО ВЖИВУЮ. VHS_VideoCombine исторически кладёт результат под ключом
    "gifs" (унаследовано от более старой версии ноды), а не "images", как у
    SaveVideo/CreateVideo в LTX/MiniMax. Пробуем оба варианта по очереди —
    после первого реального прогона стоит проверить print(outputs) и поправить,
    если структура окажется другой.
    """
    node_output = outputs["save_video"]
    video_list = node_output.get("gifs") or node_output.get("images")
    if not video_list:
        raise RuntimeError(
            f"Не нашёл видео в outputs['save_video'] — реальные ключи: {list(node_output.keys())}. "
            "Структура VHS_VideoCombine отличается от ожидаемой, поправьте result_info()."
        )
    video = video_list[0]
    return {
        "filename": video["filename"],
        "subfolder": video.get("subfolder", ""),
        "type": video.get("type", "output"),
    }


if __name__ == "__main__":
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        template = json.load(f)

    # Пример: тот же референс-кадр, что в присланном экспорте, три сегмента подряд
    workflow, job_id = build_workflow(
        template,
        prompts=[
            "The warrior cat takes the sword in both hands and raises it high above his head.",
            "The sword begins to glow red. Small red sparks circle around the cat.",
            "The blade lights up with a red flame that bursts into the sky.",
        ],
        reference_image_filename="Flux_00678_.png",  # должен уже быть загружен в ComfyUI
    )

    prompt_id = submit_job(workflow)
    print(f"Задача отправлена. job_id={job_id}, prompt_id={prompt_id}")

    outputs = wait_for_result(prompt_id)
    print("Сырые outputs узла save_video (для проверки формата):")
    print(json.dumps(outputs.get("save_video", {}), indent=2, ensure_ascii=False))

    info = result_info(outputs)
    print("Распознанная информация о видео:", info)
