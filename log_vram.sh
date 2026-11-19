#!/bin/bash
# Логирует занятую VRAM раз в секунду в CSV-файл, пока скрипт не остановят (Ctrl+C).
#
# Использование:
#   ./log_vram.sh ltx_prompt3.csv
# Запускать в ОТДЕЛЬНОМ терминале непосредственно перед отправкой генерации на сервис,
# остановить (Ctrl+C) сразу после того, как /status/{job_id} покажет "done".
#
# Результат - CSV с колонками timestamp, memory.used [MiB], memory.total [MiB] —
# удобно открыть в любой таблице и взять максимум за время прогона

OUTPUT="${1:-vram_log.csv}"

echo "Пишу в $OUTPUT. Остановить: Ctrl+C"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 1 | tee "$OUTPUT"
