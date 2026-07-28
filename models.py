"""
Каталог локальных моделей, их установка и запуск.

  python models.py list                    — что доступно и что поместится
  python models.py list --vram 48          — только влезающие в 48 ГБ
  python models.py info qwen3-32b          — подробности по модели
  python models.py hardware                — что за железо на сервере
  python models.py install qwen3-32b       — скачать и подготовить
  python models.py serve qwen3-32b         — поднять и прописать в настройки
  python models.py stop                    — остановить
  python models.py status                  — что сейчас запущено

Идея простая: администратор выбирает модель из списка в веб-интерфейсе,
всё остальное — скачивание, выбор способа запуска, параметры запуска,
прописывание адреса в настройки ассистента — делается само.

Каталог составлен под конфигурацию из двух карт RTX 4090, то есть
48 гигабайт видеопамяти суммарно. Указанные объёмы памяти — это вес
весов модели плюс запас на кэш контекста; при длинном контексте расход
растёт, поэтому для каждой модели указана рекомендуемая длина.

Способы запуска:
  vllm       — быстрый сервер с OpenAI-совместимым интерфейсом, лучший
               выбор для нескольких одновременных пользователей;
  ollama     — проще всего поставить, хорош для одиночной работы и проб;
  llama.cpp  — когда видеокарты нет вовсе или её не хватает.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config
import logging_setup

log = logging_setup.get("llm")


@dataclass
class ModelSpec:
    id: str
    title: str
    repo: str                      # репозиторий с весами
    params: str                    # размер
    arch: str
    kind: str = "llm"              # llm | embedding | reranker | vision | asr | tts
    vram_gb: float = 0             # сколько нужно видеопамяти в рекомендуемой точности
    quant: str = "AWQ int4"        # рекомендуемая точность
    context: int = 32768
    license: str = ""
    russian: str = ""              # оценка качества русского
    engine: str = "vllm"
    ollama_tag: str = ""
    notes: str = ""
    recommended: bool = False
    extra_args: list[str] = field(default_factory=list)

    def fits(self, vram_gb: float) -> bool:
        return self.vram_gb <= vram_gb * 0.92          # запас на служебные нужды


# Оценки видеопамяти даны для рекомендуемой точности и контекста; при
# увеличении контекста добавляйте примерно по 1–2 ГБ на каждые 32 тысячи
# токенов. Цифры ориентировочные — проверяйте на своём железе.
CATALOG: list[ModelSpec] = [
    # ------------------------------------------------ языковые модели ------
    ModelSpec(
        id="t-pro-2.0", title="T-Pro 2.0 (Т-Банк)", repo="t-tech/T-pro-it-2.0",
        params="32 млрд", arch="Qwen3", vram_gb=22, quant="AWQ int4", context=32768,
        license="Apache 2.0", russian="отличное — токенизатор переработан под кириллицу",
        engine="vllm", recommended=True,
        notes="Лучший выбор для русского на этой конфигурации. Построена на Qwen3, "
              "дообучена на русских данных, гибридный режим рассуждений. "
              "Заявлена вдвое большая экономичность против дистиллятов DeepSeek."),
    ModelSpec(
        id="qwen3-32b", title="Qwen3 32B", repo="Qwen/Qwen3-32B-AWQ",
        params="32 млрд", arch="Qwen3", vram_gb=21, quant="AWQ int4", context=32768,
        license="Apache 2.0", russian="хорошее", engine="vllm", ollama_tag="qwen3:32b",
        recommended=True,
        notes="Универсальная база, на ней же построена T-Pro. Берите, если нужен "
              "не только русский, но и код с многоязычием."),
    ModelSpec(
        id="qwen3-30b-a3b", title="Qwen3 30B-A3B (MoE)", repo="Qwen/Qwen3-30B-A3B",
        params="30 млрд, активны 3", arch="Qwen3 MoE", vram_gb=20, quant="AWQ int4",
        context=32768, license="Apache 2.0", russian="хорошее", engine="vllm",
        ollama_tag="qwen3:30b-a3b", recommended=True,
        notes="Смесь экспертов: отвечает заметно быстрее плотной модели того же "
              "размера, потому что на каждый токен работает лишь часть весов. "
              "Лучший вариант, если важна скорость ответа."),
    ModelSpec(
        id="gigachat3-20b-a3b", title="GigaChat3 20B-A3B (Сбер)",
        repo="ai-sage/GigaChat3-20B-A3B-instruct", params="20 млрд, активны 3",
        arch="GigaChat MoE", vram_gb=14, quant="AWQ int4", context=131072,
        license="MIT", russian="отличное — обучалась на русском изначально",
        engine="vllm",
        notes="Российская модель с открытыми весами. Длинный контекст, "
              "быстрая за счёт архитектуры смеси экспертов."),
    ModelSpec(
        id="gigachat3-10b-a1.8b", title="GigaChat3 10B-A1.8B (Сбер)",
        repo="ai-sage/GigaChat3-10B-A1.8B-instruct", params="10 млрд, активны 1,8",
        arch="GigaChat MoE", vram_gb=8, quant="AWQ int4", context=131072,
        license="MIT", russian="хорошее", engine="vllm",
        notes="Лёгкая модель: помещается на одну карту и оставляет место "
              "для эмбеддера и переранжировщика рядом."),
    ModelSpec(
        id="gemma3-27b", title="Gemma 3 27B", repo="google/gemma-3-27b-it",
        params="27 млрд", arch="Gemma 3", vram_gb=19, quant="AWQ int4", context=131072,
        license="Gemma Terms of Use (есть ограничения)", russian="среднее",
        engine="vllm", ollama_tag="gemma3:27b",
        notes="Хороша в многоязычии и работе с изображениями, но на русской "
              "технической лексике уступает Qwen и GigaChat. Внимательно "
              "прочитайте условия использования перед коммерческим применением."),
    ModelSpec(
        id="mistral-small-24b", title="Mistral Small 3.2 24B",
        repo="mistralai/Mistral-Small-3.2-24B-Instruct-2506", params="24 млрд",
        arch="Mistral", vram_gb=16, quant="AWQ int4", context=131072,
        license="Apache 2.0", russian="среднее", engine="vllm",
        ollama_tag="mistral-small:24b",
        notes="Быстрая и предсказуемая, хорошо следует инструкциям. "
              "Русский слабее, чем у Qwen3 и GigaChat."),
    ModelSpec(
        id="qwen3-14b", title="Qwen3 14B", repo="Qwen/Qwen3-14B-AWQ",
        params="14 млрд", arch="Qwen3", vram_gb=10, quant="AWQ int4", context=32768,
        license="Apache 2.0", russian="хорошее", engine="vllm", ollama_tag="qwen3:14b",
        notes="Компромисс: помещается на одну карту, оставляя вторую под "
              "эмбеддинги, распознавание речи и синтез."),
    ModelSpec(
        id="t-lite", title="T-Lite (Т-Банк)", repo="t-tech/T-lite-it-1.0",
        params="7 млрд", arch="Qwen2.5", vram_gb=6, quant="AWQ int4", context=32768,
        license="Apache 2.0", russian="хорошее для своего размера", engine="vllm",
        notes="Для слабого железа или как быстрый помощник для черновых задач."),
    ModelSpec(
        id="vikhr-nemo-12b", title="Vikhr Nemo 12B",
        repo="Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24", params="12 млрд",
        arch="Mistral Nemo", vram_gb=9, quant="AWQ int4", context=131072,
        license="Apache 2.0", russian="хорошее", engine="vllm",
        notes="Российская адаптация под русский. Линейка обновляется реже "
              "остальных — проверьте актуальность версии перед выбором."),
    ModelSpec(
        id="deepseek-r1-distill-32b", title="DeepSeek-R1 Distill Qwen 32B",
        repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", params="32 млрд",
        arch="Qwen2.5", vram_gb=21, quant="AWQ int4", context=32768,
        license="MIT", russian="хорошее", engine="vllm",
        notes="Модель с рассуждением: отвечает медленнее, зато лучше в задачах, "
              "где нужно сопоставить несколько документов. Для справочных "
              "вопросов избыточна."),
    ModelSpec(
        id="glm-4-32b", title="GLM-4 32B", repo="THUDM/glm-4-32b-0414",
        params="32 млрд", arch="GLM", vram_gb=21, quant="AWQ int4", context=131072,
        license="MIT", russian="среднее-хорошее", engine="vllm",
        notes="Альтернатива Qwen3 сопоставимого размера."),

    # ------------------------------------------------------ эмбеддинги -----
    ModelSpec(
        id="user-bge-m3", title="USER-bge-m3 (deepvk)", repo="deepvk/USER-bge-m3",
        params="0,6 млрд", arch="XLM-R", kind="embedding", vram_gb=3,
        quant="fp16", context=8192, license="Apache 2.0",
        russian="специально дообучена под русский", engine="sentence-transformers",
        recommended=True,
        notes="Основной рабочий вариант для поиска по русской документации."),
    ModelSpec(
        id="bge-m3", title="BGE-M3", repo="BAAI/bge-m3", params="0,6 млрд",
        arch="XLM-R", kind="embedding", vram_gb=3, quant="fp16", context=8192,
        license="MIT", russian="хорошее", engine="sentence-transformers",
        notes="Универсальная мультиязычная модель, умеет и плотные, и разреженные "
              "векторы одновременно."),
    ModelSpec(
        id="qwen3-embedding-4b", title="Qwen3-Embedding 4B",
        repo="Qwen/Qwen3-Embedding-4B", params="4 млрд", arch="Qwen3",
        kind="embedding", vram_gb=9, quant="fp16", context=32768,
        license="Apache 2.0", russian="очень хорошее", engine="sentence-transformers",
        notes="Заметно точнее лёгких моделей, но занимает память, которая могла бы "
              "уйти под языковую модель. Разумно, если она стоит на второй карте."),
    ModelSpec(
        id="user2-base", title="USER2-base (deepvk)", repo="deepvk/USER2-base",
        params="0,15 млрд", arch="ModernBERT", kind="embedding", vram_gb=1,
        quant="fp16", context=8192, license="Apache 2.0", russian="хорошее",
        engine="sentence-transformers",
        notes="Очень лёгкая, работает и на процессоре. Удобна, когда видеопамять "
              "целиком отдана языковой модели."),

    # ----------------------------------------------- переранжирование ------
    ModelSpec(
        id="bge-reranker-v2-m3", title="BGE-reranker v2 m3",
        repo="BAAI/bge-reranker-v2-m3", params="0,6 млрд", arch="XLM-R",
        kind="reranker", vram_gb=3, quant="fp16", context=8192, license="Apache 2.0",
        russian="хорошее", engine="sentence-transformers", recommended=True,
        notes="Пересортировывает двадцать найденных фрагментов в пятёрку лучших. "
              "Даёт ощутимый прирост точности при малых затратах."),
    ModelSpec(
        id="qwen3-reranker-4b", title="Qwen3-Reranker 4B",
        repo="Qwen/Qwen3-Reranker-4B", params="4 млрд", arch="Qwen3",
        kind="reranker", vram_gb=9, quant="fp16", context=32768, license="Apache 2.0",
        russian="очень хорошее", engine="sentence-transformers",
        notes="Точнее, но медленнее. Восьмимиллиардную версию для чата брать не "
              "стоит — задержка около пяти секунд."),

    # ------------------------------------------------------- зрение --------
    ModelSpec(
        id="qwen3-vl-8b", title="Qwen3-VL 8B", repo="Qwen/Qwen3-VL-8B-Instruct",
        params="8 млрд", arch="Qwen3-VL", kind="vision", vram_gb=10,
        quant="AWQ int4", context=32768, license="Apache 2.0",
        russian="единственная не путающая кириллицу с латиницей", engine="vllm",
        recommended=True,
        notes="Описание фотографий продукции и чтение надписей с шильдиков."),
    ModelSpec(
        id="qwen3-vl-32b", title="Qwen3-VL 32B", repo="Qwen/Qwen3-VL-32B-Instruct",
        params="32 млрд", arch="Qwen3-VL", kind="vision", vram_gb=22,
        quant="AWQ int4", context=32768, license="Apache 2.0", russian="отличное",
        engine="vllm",
        notes="Заметно точнее на сложных сканах и чертежах. Имеет смысл запускать "
              "разово для обработки архива, а не держать постоянно."),

    # --------------------------------------------------------- речь -------
    ModelSpec(
        id="gigaam-v3", title="GigaAM v3 (Сбер)", repo="salute-developers/GigaAM",
        params="0,24 млрд", arch="Conformer RNNT", kind="asr", vram_gb=2,
        quant="fp16", context=0, license="MIT", russian="лучшее на чистой речи",
        engine="native", recommended=True,
        notes="Распознавание русской речи. На шумных записях уступает Whisper, "
              "на студийных — выигрывает."),
    ModelSpec(
        id="whisper-large-v3-turbo", title="Whisper large-v3-turbo",
        repo="openai/whisper-large-v3-turbo", params="0,8 млрд", arch="Whisper",
        kind="asr", vram_gb=4, quant="int8", context=0, license="MIT",
        russian="хорошее, устойчиво к шуму", engine="faster-whisper", recommended=True,
        notes="Универсальный вариант. Запускать через faster-whisper — быстрее "
              "и экономнее оригинальной реализации."),
    ModelSpec(
        id="t-one", title="T-one (Т-Банк)", repo="t-tech/T-one",
        params="0,07 млрд", arch="Conformer streaming", kind="asr", vram_gb=1,
        quant="fp16", context=0, license="Apache 2.0",
        russian="лучшее на телефонии 8 кГц", engine="native",
        notes="Потоковое распознавание с задержкой около трети секунды. "
              "Нужна именно для телефонии."),
    ModelSpec(
        id="silero-tts-v4", title="Silero TTS v4 (русский)",
        repo="snakers4/silero-models", params="—", arch="Silero", kind="tts",
        vram_gb=1, quant="fp16", context=0, license="свободная для большинства случаев",
        russian="хорошее, несколько дикторов", engine="native", recommended=True,
        notes="Работает на процессоре, лицензия допускает использование в компании. "
              "В отличие от XTTS и Fish Speech, которые распространяются "
              "по некоммерческим лицензиям."),
]

BY_ID = {m.id: m for m in CATALOG}


# ---------------------------------------------------------------- железо ----
def hardware() -> dict:
    """Что за сервер: карты, память, процессор, диск."""
    info: dict = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import multiprocessing
        info["cpu_cores"] = multiprocessing.cpu_count()
    except Exception:  # noqa: BLE001
        info["cpu_cores"] = 0
    info["cpu_model"] = _cpu_model()
    info["ram_gb"] = round(_total_ram() / 1024 ** 3, 1)
    info["disk_free_gb"] = round(shutil.disk_usage(str(config.DATA_DIR)).free / 1024 ** 3, 1)
    info["gpus"] = gpus()
    info["vram_total_gb"] = round(sum(g["memory_total_gb"] for g in info["gpus"]), 1)
    info["engines"] = {
        "vllm": _has_python_module("vllm"),
        "ollama": bool(shutil.which("ollama")),
        "llama.cpp": bool(shutil.which("llama-server")),
        "docker": bool(shutil.which("docker")),
        "huggingface-cli": bool(shutil.which("huggingface-cli") or shutil.which("hf")),
    }
    return info


def _cpu_model() -> str:
    try:
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or "неизвестно"


def _total_ram() -> int:
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        if platform.system() == "Darwin":
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                      capture_output=True, text=True, timeout=5).stdout)
    except Exception:  # noqa: BLE001
        pass
    return 0


def gpus() -> list[dict]:
    """Список видеокарт через nvidia-smi. Пустой список — карт нет."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,"
             "utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout
    except Exception:  # noqa: BLE001
        return []
    cards = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            cards.append({
                "index": int(parts[0]), "name": parts[1],
                "memory_total_gb": round(float(parts[2]) / 1024, 1),
                "memory_used_gb": round(float(parts[3]) / 1024, 1),
                "utilization": float(parts[4]) if len(parts) > 4 and parts[4] != "[N/A]" else 0,
                "temperature": float(parts[5]) if len(parts) > 5 and parts[5] != "[N/A]" else 0,
                "power_w": float(parts[6]) if len(parts) > 6 and parts[6] != "[N/A]" else 0,
            })
        except ValueError:
            continue
    return cards


def _has_python_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


# ---------------------------------------------------------------- каталог ---
def catalog(kind: str | None = None, vram_gb: float | None = None) -> list[dict]:
    """Каталог с пометкой, поместится ли модель в имеющуюся видеопамять."""
    if vram_gb is None:
        hw = hardware()
        vram_gb = hw["vram_total_gb"] or 0
    out = []
    for m in CATALOG:
        if kind and m.kind != kind:
            continue
        item = asdict(m)
        item["fits"] = m.fits(vram_gb) if vram_gb else None
        item["installed"] = is_installed(m)
        out.append(item)
    return out


def models_dir() -> Path:
    path = Path(config.MODELS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def local_path(spec: ModelSpec) -> Path:
    return models_dir() / spec.id


# Пока идёт загрузка, рядом с весами лежит этот файл. Он и отличает
# «скачано полностью» от «скачано наполовину»: раньше признаком готовности
# был любой файл .safetensors в папке, поэтому оборванная на середине
# загрузка (перезапуск процесса, разрыв связи) выглядела как готовая
# модель — и следующий запуск поднимал сервер на неполных весах.
DOWNLOADING = ".загрузка-не-завершена"


def download_in_progress(spec: ModelSpec) -> bool:
    return (local_path(spec) / DOWNLOADING).exists()


def is_installed(spec: ModelSpec) -> bool:
    path = local_path(spec)
    if download_in_progress(spec):
        return False
    if path.exists() and any(path.rglob("*.safetensors")):
        return True
    if spec.ollama_tag and shutil.which("ollama"):
        try:
            out = subprocess.run(["ollama", "list"], capture_output=True,
                                 text=True, timeout=15).stdout
            return spec.ollama_tag.split(":")[0] in out
        except Exception:  # noqa: BLE001
            return False
    return False


# -------------------------------------------------------------- установка ---
def install(model_id: str, engine: str | None = None, progress=None) -> dict:
    """Скачивает веса. Возвращает отчёт; сообщения идут в журнал."""
    spec = BY_ID.get(model_id)
    if not spec:
        raise ValueError(f"нет такой модели в каталоге: {model_id}")
    engine = engine or spec.engine
    say = progress or (lambda text: log.info("%s", text))

    hw = hardware()
    if spec.kind == "llm" and hw["vram_total_gb"] and not spec.fits(hw["vram_total_gb"]):
        say(f"Внимание: модели нужно около {spec.vram_gb} ГБ видеопамяти, "
            f"а на сервере {hw['vram_total_gb']} ГБ. Запуск может не удаться.")

    if engine == "ollama" and spec.ollama_tag:
        if not shutil.which("ollama"):
            raise RuntimeError("не установлен ollama — поставьте его или выберите vllm")
        say(f"Загружаю через ollama: {spec.ollama_tag}")
        proc = subprocess.Popen(["ollama", "pull", spec.ollama_tag],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:               # type: ignore[union-attr]
            say(line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ollama вернул код {proc.returncode}")
        return {"model": spec.id, "engine": "ollama", "tag": spec.ollama_tag}

    target = local_path(spec)
    target.mkdir(parents=True, exist_ok=True)
    marker = target / DOWNLOADING
    if marker.exists():
        say("Найдена незавершённая загрузка — продолжаю с того же места.")
    marker.write_text(
        "Загрузка весов не завершена. Пока этот файл на месте, модель\n"
        "считается неготовой и запустить её нельзя: сервер на неполных\n"
        "весах падает не сразу, а на первом же нестандартном запросе.\n"
        "Файл удаляется сам после успешной загрузки.\n", encoding="utf-8")
    cli = shutil.which("hf") or shutil.which("huggingface-cli")
    if cli:
        say(f"Скачиваю веса {spec.repo} в {target}")
        cmd = [cli, "download", spec.repo, "--local-dir", str(target)]
        env = dict(os.environ)
        if config.HF_MIRROR:
            env["HF_ENDPOINT"] = config.HF_MIRROR
            say(f"Использую зеркало: {config.HF_MIRROR}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env)
        for line in proc.stdout:               # type: ignore[union-attr]
            say(line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"загрузка не удалась, код {proc.returncode}. "
                               "Из России площадка недоступна напрямую — укажите "
                               "зеркало в настройке HF_MIRROR или используйте прокси.")
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "нет ни huggingface-cli, ни библиотеки huggingface_hub: "
                "pip install huggingface_hub") from exc
        say(f"Скачиваю веса {spec.repo}")
        snapshot_download(spec.repo, local_dir=str(target),
                          endpoint=config.HF_MIRROR or None)
    # Признак незавершённости снимаем только теперь, когда всё скачано.
    marker.unlink(missing_ok=True)
    say("Готово.")
    return {"model": spec.id, "engine": engine, "path": str(target)}


# ------------------------------------------------------------------ запуск --
def _pid_file() -> Path:
    return Path(config.DATA_DIR) / "model_server.json"


def status() -> dict:
    path = _pid_file()
    if not path.exists():
        return {"running": False}
    try:
        state = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {"running": False}
    pid = state.get("pid")
    alive = False
    if pid:
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
    state["running"] = alive
    if alive:
        state["uptime_seconds"] = int(time.time() - state.get("started", time.time()))
    return state


def serve(model_id: str, engine: str | None = None, port: int | None = None,
          apply_config: bool = True) -> dict:
    """Поднимает сервер модели и, если попросили, прописывает его в настройки."""
    spec = BY_ID.get(model_id)
    if not spec:
        raise ValueError(f"нет такой модели: {model_id}")
    engine = engine or spec.engine
    port = port or config.LOCAL_MODEL_PORT
    stop()

    if engine == "ollama":
        if not shutil.which("ollama"):
            raise RuntimeError("не установлен ollama")
        proc = subprocess.Popen(["ollama", "serve"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base_url = "http://127.0.0.1:11434/v1"
        served = spec.ollama_tag
    elif engine == "vllm":
        if not _has_python_module("vllm"):
            raise RuntimeError("не установлен vllm: pip install vllm")
        path = local_path(spec)
        if download_in_progress(spec):
            raise RuntimeError(
                "загрузка весов этой модели не завершена. Запуск на неполных "
                "весах даёт сервер, который падает не сразу, а на первом же "
                "нестандартном запросе. Скачайте модель заново.")
        source = str(path) if path.exists() and any(path.rglob("*.safetensors")) else spec.repo
        cards = len(gpus()) or 1
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", source,
               "--served-model-name", spec.id,
               "--port", str(port),
               "--max-model-len", str(min(spec.context, config.LOCAL_MODEL_CONTEXT)),
               "--gpu-memory-utilization", str(config.LOCAL_MODEL_GPU_FRACTION)]
        if cards > 1:
            cmd += ["--tensor-parallel-size", str(cards)]
        if "AWQ" in spec.quant:
            cmd += ["--quantization", "awq"]
        cmd += spec.extra_args
        log.info("запускаю vllm: %s", " ".join(cmd))
        logs = Path(config.LOG_DIR) / "model_server.log"
        logs.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(cmd, stdout=logs.open("ab"), stderr=subprocess.STDOUT)
        base_url = f"http://127.0.0.1:{port}/v1"
        served = spec.id
    else:
        raise RuntimeError(f"запуск через «{engine}» из интерфейса не поддерживается; "
                           "смотрите документацию по ручной установке")

    state = {"pid": proc.pid, "model": spec.id, "engine": engine,
             "base_url": base_url, "served_name": served, "started": time.time()}
    _pid_file().write_text(json.dumps(state, ensure_ascii=False))

    if apply_config:
        import webui
        updates: dict[str, str] = {}
        if spec.kind == "llm":
            # Провайдер local сам находит адрес запущенного сервера, поэтому
            # прописывать порт в настройки не нужно: после перезапуска модели
            # на другом порту ничего не сломается.
            updates["LLM_PROVIDER"] = "local"
            updates["LOCAL_LLM_MODEL"] = served
        elif spec.kind == "vision":
            updates["VISION_PROVIDER"] = "openai"
            updates["VISION_MODEL"] = served
            updates["OPENAI_BASE_URL"] = base_url
            updates["OPENAI_API_KEY"] = "local"
        elif spec.kind == "embedding":
            updates["EMBEDDINGS_PROVIDER"] = "openai"
            updates["EMBEDDINGS_MODEL"] = served
            updates["OPENAI_BASE_URL"] = base_url
            updates["OPENAI_API_KEY"] = "local"
        if updates:
            webui.write_env(updates)
        try:
            import llm as llm_mod
            llm_mod.reset()
        except Exception:  # noqa: BLE001
            pass
        log.info("настройки обновлены: ассистент будет обращаться к %s", base_url)

    log.info("модель «%s» запускается, процесс %d", spec.title, proc.pid)
    return state


def check(model_id: str) -> dict:
    """
    Что мешает запустить эту модель на этой машине.

    Отдельная проверка нужна, потому что запуск модели — операция на
    десятки минут, и узнавать о нехватке видеопамяти или отсутствии vllm
    через полчаса ожидания неправильно.
    """
    spec = BY_ID.get(model_id)
    if not spec:
        return {"ok": False, "problems": [f"нет такой модели: {model_id}"]}
    hw = hardware()
    problems: list[str] = []
    notes: list[str] = []

    if spec.engine == "vllm" and not _has_python_module("vllm"):
        problems.append("не установлен vllm (pip install vllm) — "
                        "он и запускает модель")
    if spec.engine == "ollama" and not shutil.which("ollama"):
        problems.append("не установлен ollama")

    vram = hw.get("vram_total_gb") or 0
    if vram == 0:
        problems.append("видеокарт не найдено: модель такого размера на "
                        "процессоре будет отвечать минутами")
    elif spec.vram_gb > vram:
        problems.append(f"нужно {spec.vram_gb} ГБ видеопамяти, доступно {vram} ГБ")
    elif spec.vram_gb > vram * config.LOCAL_MODEL_GPU_FRACTION:
        notes.append(f"модель займёт почти всю видеопамять ({spec.vram_gb} из "
                     f"{vram} ГБ) — на зрение и эмбеддинги места не останется")

    free = hw.get("disk_free_gb") or 0
    need = max(spec.vram_gb * 1.2, 5)
    if not is_installed(spec) and free < need:
        problems.append(f"на диске {free} ГБ, для весов нужно около {need:.0f} ГБ")

    if not is_installed(spec):
        notes.append("весов ещё нет — они скачаются при установке "
                     f"(около {spec.vram_gb} ГБ)")
    if config.HF_MIRROR:
        notes.append(f"скачивание пойдёт через зеркало {config.HF_MIRROR}")
    else:
        notes.append("площадка с весами из России напрямую недоступна: "
                     "укажите HF_MIRROR или прокси, иначе загрузка не пойдёт")
    return {"ok": not problems, "model": spec.id, "title": spec.title,
            "problems": problems, "notes": notes,
            "vram_needed": spec.vram_gb, "vram_available": vram,
            "installed": is_installed(spec), "engine": spec.engine}


def stop() -> bool:
    state = status()
    if not state.get("running"):
        return False
    try:
        os.kill(state["pid"], 15)
        log.info("остановлен процесс модели %d", state["pid"])
    except OSError as exc:
        log.warning("не удалось остановить процесс: %s", exc)
        return False
    _pid_file().unlink(missing_ok=True)
    return True


def wait_ready(base_url: str, timeout: int = 600) -> bool:
    """Ждёт, пока сервер модели начнёт отвечать. Загрузка весов долгая."""
    import httpx
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(base_url.replace("/v1", "") + "/v1/models", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    return False


# --------------------------------------------------------------------- CLI --
def main() -> int:
    p = argparse.ArgumentParser(description="Локальные модели")
    p.add_argument("command", choices=["list", "info", "hardware", "install",
                                       "serve", "stop", "status"])
    p.add_argument("model", nargs="?")
    p.add_argument("--kind", choices=["llm", "embedding", "reranker", "vision", "asr", "tts"])
    p.add_argument("--vram", type=float)
    p.add_argument("--engine")
    p.add_argument("--port", type=int)
    a = p.parse_args()
    logging_setup.setup()

    if a.command == "hardware":
        hw = hardware()
        print(f"Система     : {hw['platform']}")
        print(f"Процессор   : {hw['cpu_model']} ({hw['cpu_cores']} ядер)")
        print(f"Память      : {hw['ram_gb']} ГБ")
        print(f"Диск свободн: {hw['disk_free_gb']} ГБ")
        if hw["gpus"]:
            for g in hw["gpus"]:
                print(f"Видеокарта {g['index']}: {g['name']}, {g['memory_total_gb']} ГБ "
                      f"(занято {g['memory_used_gb']} ГБ, загрузка {g['utilization']}%)")
            print(f"Видеопамять суммарно: {hw['vram_total_gb']} ГБ")
        else:
            print("Видеокарты не обнаружены — локальные модели пойдут только на процессоре")
        print("Движки      : " + ", ".join(f"{k}: {'есть' if v else 'нет'}"
                                            for k, v in hw["engines"].items()))
        return 0

    if a.command == "list":
        vram = a.vram
        if vram is None:
            vram = hardware()["vram_total_gb"]
        print(f"Доступная видеопамять: {vram or 'не определена'} ГБ\n")
        for item in catalog(a.kind, vram):
            fits = "" if item["fits"] is None else ("  " if item["fits"] else "  ✗ не влезет")
            star = " ★" if item["recommended"] else "  "
            installed = " [установлена]" if item["installed"] else ""
            print(f"{star}{item['id']:<26} {item['params']:<18} "
                  f"{item['vram_gb']:>4} ГБ  {item['kind']:<9}{fits}{installed}")
            print(f"    {item['title']} · {item['license']} · русский: {item['russian']}")
        print("\n★ — рекомендуется. Подробности: python models.py info <id>")
        return 0

    if a.command == "info":
        spec = BY_ID.get(a.model or "")
        if not spec:
            print("Укажите идентификатор модели из списка.")
            return 1
        d = asdict(spec)
        print(f"{spec.title}\n")
        print(f"Идентификатор : {spec.id}")
        print(f"Репозиторий   : {spec.repo}")
        print(f"Размер        : {spec.params}, архитектура {spec.arch}")
        print(f"Видеопамять   : около {spec.vram_gb} ГБ при {spec.quant}")
        print(f"Контекст      : {spec.context} токенов")
        print(f"Лицензия      : {spec.license}")
        print(f"Русский язык  : {spec.russian}")
        print(f"Запуск через  : {spec.engine}")
        print(f"Установлена   : {'да' if is_installed(spec) else 'нет'}")
        print(f"\n{spec.notes}")
        return 0

    if a.command == "install":
        print(json.dumps(install(a.model, a.engine, progress=print),
                         ensure_ascii=False, indent=2))
    elif a.command == "serve":
        state = serve(a.model, a.engine, a.port)
        print(f"Запущено: {state['model']} через {state['engine']}, "
              f"адрес {state['base_url']}")
        print("Загрузка весов занимает несколько минут. Проверка: python models.py status")
    elif a.command == "stop":
        print("Остановлено." if stop() else "Ничего не запущено.")
    elif a.command == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
