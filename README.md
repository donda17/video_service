# Video Director Service

Сервис для генерации видео через ComfyUI (LTX 2.3 Director + MiniMax H3) с веб-интерфейсом.

## Структура

- `comfy_client.py` - общая часть: отправка задач в ComfyUI, ожидание результата, загрузка картинок.
- `ltx_service.py` - адаптер под LTX 2.3 Director (список сцен → таймлайн).
- `minimax_service.py` - адаптер под MiniMax H3 (один промпт, 4–15 сек).
- `app.py` - FastAPI-сервер: очередь на одну GPU, эндпоинты `/generate`, `/generate-minimax`, `/upload-image`, `/status/{job_id}`.
- `index.html` - фронтенд (открывается сервером на `/`).
- `LTX_API.json`, `Minimax_h3_t2v_i2v_api.json` - экспортированные и починенные API-воркфлоу ComfyUI.

## Установка на сервере (Linux)

```bash
git clone <адрес-этого-репозитория>
cd <папка-репозитория>

python3 -m venv venv
source venv/bin/activate

# PyTorch - под вашу версию CUDA (проверить: nvidia-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

pip install -r requirements.txt
```

### ComfyUI и модели

ComfyUI ставится отдельно (не в этом репозитории - там гигабайтные веса):

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git ~/ComfyUI
cd ~/ComfyUI
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git
git clone https://github.com/city96/ComfyUI-GGUF.git
git clone https://github.com/kijai/ComfyUI-KJNodes.git
git clone https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git
```

**Модели под LTX 2.3 Director** (раскладываются по `~/ComfyUI/models/...`):

| Файл | Папка |
|---|---|
| `ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf` | `unet/` |
| `gemma-3-12b-it-qat-UD-Q3_K_XL.gguf` | `text_encoders/` (или `clip/`, если модель не появляется в списке) |
| `ltx-2.3_text_projection_bf16.safetensors` | `text_encoders/` |
| `LTX23_video_vae_bf16.safetensors` | `vae/` |
| `LTX23_audio_vae_bf16.safetensors` | `vae/` |
| `taeltx2_3.safetensors` | `vae/` |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `latent_upscale_models/` |

Источники: https://github.com/wildminder/awesome-ltx2, https://huggingface.co/Kijai/LTX2.3_comfy, https://huggingface.co/QuantStack/LTX-2.3-GGUF, https://huggingface.co/unsloth/gemma-3-12b-it-qat-GGUF

**Модели под MiniMax H3** (проще через сам ComfyUI: Workflow → Browse Templates → Video → MiniMax H3, скачивание встроено; либо см. `Comfy-Org/MiniMax-H3` на HuggingFace вручную).

### Запуск ComfyUI

```bash
cd ~/ComfyUI
source venv/bin/activate
python main.py --listen 127.0.0.1 --port 8188
```

### Запуск сервиса

```bash
source venv/bin/activate   # venv репозитория, не ComfyUI
export LTX_API_KEY="придумайте_длинный_случайный_ключ"   # опционально, но рекомендуется
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Откройте `http://<адрес-машины>:8000` (или через SSH-туннель `ssh -L 8000:127.0.0.1:8000 ...`).

Если задавали `LTX_API_KEY` - впишите тот же ключ в `index.html`, в константу `API_KEY` в начале `<script>`.

## Ограничения

- LTX Director: до 60 сек суммарно, до 12 сцен за раз (`app.py`, `MAX_TOTAL_DURATION_SEC` / `MAX_SCENES`).
- MiniMax H3: 4–15 сек на генерацию - жёсткий лимит самой модели.
- Очередь: одна GPU - задачи выполняются строго последовательно, независимо от того, какую модель выбрал пользователь.

## Публикация наружу

Порты ComfyUI (8188) и сервиса (8000) слушают только `127.0.0.1` - наружу ничего не торчит напрямую.
Для доступа снаружи - Cloudflare Tunnel:

```bash
cloudflared tunnel login
cloudflared tunnel create video-service
# настроить ~/.cloudflared/config.yml: hostname → http://127.0.0.1:8000
cloudflared tunnel route dns video-service ваш-поддомен.com
sudo cloudflared service install
```
