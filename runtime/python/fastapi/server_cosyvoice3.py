from __future__ import annotations

import io
import logging
import os
import queue
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from tts_engine import TTSEngine, create_engine, decode_base64_audio, pcm_bytes_to_wav_bytes
from normalizer import create_normalizer_pipeline, NormalizerPipeline

SAMPLE_RATE = 24000

API_MODEL_NAME = "optimized_short"
COSYVOICE_ROOT = Path(__file__).resolve().parents[3]
COSYVOICE_MODEL_DIR = os.environ.get("COSYVOICE_MODEL_DIR", "")
DEFAULT_LANG = os.environ.get("QWEN_DEFAULT_LANG", "ru").strip().lower() or "ru"
DEFAULT_PROMPT_AUDIO = os.environ.get(
    "COSYVOICE_DEFAULT_REF_AUDIO",
    str(COSYVOICE_ROOT / "asset" / "qwen_ref_4.wav"),
)
COSYVOICE_COMPAT_MODE = os.environ.get("COSYVOICE_COMPAT_MODE", "cross_lingual").strip() or "cross_lingual"
COSYVOICE_DEFAULT_SPEED_RAW = os.environ.get("COSYVOICE_DEFAULT_SPEED")
COSYVOICE_DEFAULT_SPEED: float | None = None
COSYVOICE_DEFAULT_SPEAKER = os.environ.get("COSYVOICE_DEFAULT_SPEAKER", "").strip() or None
COSYVOICE_DEFAULT_INSTRUCT = os.environ.get("COSYVOICE_DEFAULT_INSTRUCT", "").strip() or None
COSYVOICE_PROMPT_TEXT = os.environ.get("COSYVOICE_PROMPT_TEXT", "").strip() or None
COSYVOICE_BACKEND = os.environ.get("COSYVOICE_BACKEND", "native").lower().strip()
COSYVOICE_FP16 = os.environ.get("COSYVOICE_FP16", "true").lower() in ("1", "true", "yes")
COSYVOICE_LOAD_TRT = os.environ.get("COSYVOICE_LOAD_TRT", "false").lower() in ("1", "true", "yes")
COSYVOICE_TRT_ENGINE_DIR = os.environ.get("COSYVOICE_TRT_ENGINE_DIR", "")
COSYVOICE_TRT_SERVE_URL = os.environ.get("COSYVOICE_TRT_SERVE_URL", "")
COSYVOICE_HF_MODEL_DIR = os.environ.get("COSYVOICE_HF_MODEL_DIR", "")
COSYVOICE_GPU_MEM = float(os.environ.get("COSYVOICE_GPU_MEM", "0.4"))
TTS_PORT = int(os.environ.get("TTS_PORT", "8000"))
TTS_HOST = os.environ.get("TTS_HOST", "0.0.0.0")
TTS_WARMUP_ENABLED = os.environ.get("TTS_WARMUP_ENABLED", "true").lower() in ("1", "true", "yes")
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", "5000"))
ENABLE_TEXT_NORMALIZATION = os.environ.get("ENABLE_TEXT_NORMALIZATION", "true").lower() in ("1", "true", "yes")
GENERATED_AUDIO_DIR = Path(
    os.environ.get("GENERATED_AUDIO_DIR", str(Path(__file__).resolve().parent / "generated"))
).expanduser()

LANGUAGE_MAP = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "auto": "Auto",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cosyvoice3_tts.server")

engine: TTSEngine | None = None
normalizer_pipeline: NormalizerPipeline | None = None
model_status = "not_loaded"
model_error: str | None = None
warmup_completed = False

audio_save_queue: queue.Queue[tuple[str, bytes]] = queue.Queue()
audio_save_worker_started = False

MAX_LOG_TEXT = 300

HTTP_REQUESTS_TOTAL = Counter(
    "tts_http_requests_total",
    "Total HTTP requests by method, normalized path, and status",
    ["method", "path", "status"],
)
HTTP_REQUEST_DURATION_MS = Histogram(
    "tts_http_request_duration_ms",
    "HTTP request duration in milliseconds",
    ["method", "path"],
    buckets=(5, 10, 25, 50, 100, 200, 400, 800, 1500, 3000, 5000, 10000, 20000),
)
TTS_REQUESTS_TOTAL = Counter(
    "tts_request_total",
    "Total TTS requests by endpoint, mode, language, and status",
    ["endpoint", "mode", "lang", "status"],
)
TTS_ACTIVE_REQUESTS = Gauge(
    "tts_active_requests",
    "Current active TTS requests",
    ["endpoint", "mode"],
)
TTS_ACTIVE_STREAMS = Gauge(
    "tts_active_streams",
    "Current active TTS streams",
    ["endpoint", "mode"],
)
TTS_STREAM_FIRST_CHUNK_MS = Histogram(
    "tts_first_chunk_ms",
    "Time from request start to first audio chunk in milliseconds",
    ["endpoint", "mode", "lang"],
    buckets=(10, 25, 50, 100, 200, 400, 800, 1500, 3000, 5000, 10000, 20000),
)
TTS_FIRST_CHUNK_RTF = Histogram(
    "tts_first_chunk_rtf",
    "Real-time factor measured at first audio chunk",
    ["endpoint", "mode", "lang"],
    buckets=(0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
)
TTS_AUDIO_DURATION_SEC = Histogram(
    "tts_audio_duration_sec",
    "Generated audio duration in seconds",
    ["endpoint", "mode", "lang", "status"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60),
)
TTS_STREAM_COMPLETE_MS = Histogram(
    "tts_e2e_ms",
    "Time from request start to full audio completion in milliseconds",
    ["endpoint", "mode", "lang", "status"],
    buckets=(10, 25, 50, 100, 200, 400, 800, 1500, 3000, 5000, 10000, 20000, 30000, 60000),
)
TTS_STREAM_RTF = Histogram(
    "tts_rtf",
    "Real-time factor for completed TTS requests",
    ["endpoint", "mode", "lang", "status"],
    buckets=(0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
)


def observe_duration_ms(metric: Histogram, labels: tuple[str, ...], duration_seconds: float) -> None:
    metric.labels(*labels).observe(duration_seconds * 1000.0)


def normalize_http_path(req: Request) -> str:
    route = req.scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return str(path)
    return req.url.path


class TTSStreamRequest(BaseModel):
    text: str
    lang: str | None = DEFAULT_LANG
    language: str | None = None
    speaker: str | None = None
    instruct: str | None = None
    sample_rate_hz: int | None = None
    codec: str | None = None
    bitrate: int | None = None
    bandwidth_hz_est: float | None = None
    snr_db: float | None = None
    silence_ratio: float | None = None
    rms_dbfs: float | None = None
    packet_loss_pct: float | None = None
    jitter_ms: float | None = None
    barge_in: bool = False


class OpenAISpeechRequest(BaseModel):
    input: str
    model: str = API_MODEL_NAME
    voice: str | None = None
    response_format: str = "pcm"
    stream: bool = True
    lang: str = DEFAULT_LANG
    instruct: str | None = None


def resolve_lang(lang: str | None) -> str:
    resolved = (lang or DEFAULT_LANG).strip().lower()
    if resolved not in LANGUAGE_MAP:
        raise HTTPException(status_code=400, detail=f"Unsupported lang: {resolved}")
    return resolved


def get_request_lang(req: TTSStreamRequest) -> str:
    return req.lang or req.language or DEFAULT_LANG


def short_text_for_log(text: str, max_len: int = MAX_LOG_TEXT) -> str:
    normalized = text.replace("\n", " ").strip()
    return normalized if len(normalized) <= max_len else normalized[:max_len] + "... [truncated]"


_COSYVOICE3_PREFIX = "You are a helpful assistant.<|endofprompt|>"


def ensure_cv3_prefix(text: str) -> str:
    if "<|endofprompt|>" in text:
        return text
    return _COSYVOICE3_PREFIX + text


def current_sample_rate() -> int:
    if engine is None:
        return SAMPLE_RATE
    candidates = [engine]
    for attr in ("cosyvoice", "t2w"):
        value = getattr(engine, attr, None)
        if value is not None:
            candidates.append(value)
    for candidate in candidates:
        for attr in ("sample_rate", "sampling_rate"):
            value = getattr(candidate, attr, None)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
    return SAMPLE_RATE


def get_default_speed() -> float:
    if COSYVOICE_DEFAULT_SPEED is None:
        return 1.0
    return COSYVOICE_DEFAULT_SPEED


def get_compat_mode() -> str:
    return COSYVOICE_COMPAT_MODE


def get_request_speaker(speaker: str | None = None) -> str | None:
    return (speaker or COSYVOICE_DEFAULT_SPEAKER or "").strip() or None


def get_request_instruct(instruct: str | None = None) -> str | None:
    return (instruct or COSYVOICE_DEFAULT_INSTRUCT or "").strip() or None


def load_default_prompt_audio() -> str | None:
    if not DEFAULT_PROMPT_AUDIO:
        return None
    path = Path(DEFAULT_PROMPT_AUDIO).expanduser()
    if not path.is_file():
        return None
    return str(path)


def slugify_text(text: str, max_len: int = 48) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ]+", "_", normalized)
    normalized = normalized.strip("_")
    if not normalized:
        return "empty"
    return normalized[:max_len].rstrip("_") or "empty"


def build_audio_output_path(text: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slug = slugify_text(text)
    return GENERATED_AUDIO_DIR / f"{timestamp}_{slug}.wav"


def audio_save_worker() -> None:
    while True:
        path_str, pcm_bytes = audio_save_queue.get()
        try:
            wav_bytes = pcm_bytes_to_wav_bytes(pcm_bytes, sample_rate=current_sample_rate())
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)
            with open(path_str, "wb") as f:
                f.write(wav_bytes)
            logger.info("Saved generated audio to %s", path_str)
        except Exception:
            logger.exception("Failed to save generated audio to %s", path_str)
        finally:
            audio_save_queue.task_done()


def ensure_audio_save_worker() -> None:
    global audio_save_worker_started
    if audio_save_worker_started:
        return
    worker = threading.Thread(target=audio_save_worker, name="audio-save", daemon=True)
    worker.start()
    audio_save_worker_started = True


def resolve_prompt_audio(
    prompt_audio_b64: str | None = None,
    prompt_audio_file: UploadFile | None = None,
) -> str | io.BytesIO | None:
    if prompt_audio_b64:
        return decode_base64_audio(prompt_audio_b64)
    if prompt_audio_file is not None:
        # Save to temp file so torchaudio can read it multiple times
        import tempfile
        suffix = ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(prompt_audio_file.file.read())
            return tmp.name
    return None


def generate_audio_stream(
    text: str,
    mode: str,
    lang: str,
    speaker: str | None,
    instruct: str | None,
    prompt_text: str | None,
    prompt_audio: str | io.BytesIO | None,
    source_audio: str | io.BytesIO | None,
    speed: float,
    req_id: str,
    endpoint: str,
) -> Generator[bytes, None, None]:
    output_path = build_audio_output_path(text) if str(GENERATED_AUDIO_DIR) else None
    started = time.perf_counter()
    first_chunk_at = None
    total_bytes = 0
    chunk_count = 0
    saved_chunks: list[bytes] = []
    status = "ok"
    TTS_ACTIVE_REQUESTS.labels(endpoint, mode).inc()
    TTS_ACTIVE_STREAMS.labels(endpoint, mode).inc()

    logger.info(
        "[%s] stream_start | mode=%s | lang=%s | speaker=%s | chars=%s",
        req_id, mode, lang, speaker, len(text),
    )

    try:
        for chunk in engine.stream_generate(
            text=text,
            mode=mode,
            speaker=speaker,
            instruct=instruct,
            prompt_text=prompt_text,
            prompt_audio=prompt_audio,
            source_audio=source_audio,
            speed=speed,
        ):
            chunk_count += 1
            total_bytes += len(chunk)
            saved_chunks.append(chunk)
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
                sample_rate = current_sample_rate()
                first_chunk_audio_sec = len(chunk) / 2.0 / sample_rate if chunk else 0.0
                first_chunk_latency_sec = first_chunk_at - started
                first_chunk_rtf = (
                    first_chunk_latency_sec / first_chunk_audio_sec if first_chunk_audio_sec > 0 else 0.0
                )
                logger.info(
                    "[%s] first_chunk | bytes=%s | audio_ms=%.0f | latency_ms=%.0f | rtf=%.2f",
                    req_id, len(chunk), first_chunk_audio_sec * 1000.0,
                    first_chunk_latency_sec * 1000.0, first_chunk_rtf,
                )
                observe_duration_ms(
                    TTS_STREAM_FIRST_CHUNK_MS,
                    (endpoint, mode, lang),
                    first_chunk_latency_sec,
                )
                if first_chunk_audio_sec > 0:
                    TTS_FIRST_CHUNK_RTF.labels(endpoint, mode, lang).observe(first_chunk_rtf)
            yield chunk
    except Exception as exc:
        status = "error"
        logger.exception("[%s] stream error: %s", req_id, exc)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}") from exc
    finally:
        elapsed = time.perf_counter() - started
        sample_rate = current_sample_rate()
        audio_sec = total_bytes / 2.0 / sample_rate if total_bytes else 0.0
        rtf = elapsed / audio_sec if audio_sec > 0 else 0.0
        if saved_chunks and output_path:
            audio_save_queue.put((str(output_path), b"".join(saved_chunks)))
        observe_duration_ms(
            TTS_STREAM_COMPLETE_MS,
            (endpoint, mode, lang, status),
            elapsed,
        )
        TTS_REQUESTS_TOTAL.labels(endpoint, mode, lang, status).inc()
        TTS_ACTIVE_REQUESTS.labels(endpoint, mode).dec()
        TTS_ACTIVE_STREAMS.labels(endpoint, mode).dec()
        if audio_sec > 0:
            TTS_AUDIO_DURATION_SEC.labels(endpoint, mode, lang, status).observe(audio_sec)
            TTS_STREAM_RTF.labels(endpoint, mode, lang, status).observe(rtf)
        logger.info(
            "[%s] stream_done | chunks=%s | bytes=%s | total_ms=%.0f | rtf=%.2f",
            req_id, chunk_count, total_bytes, elapsed * 1000.0, rtf,
        )


def fetch_audio_bytes(
    text: str,
    mode: str,
    lang: str,
    speaker: str | None,
    instruct: str | None,
    prompt_text: str | None,
    prompt_audio: str | io.BytesIO | None,
    source_audio: str | io.BytesIO | None,
    speed: float,
    req_id: str,
    endpoint: str,
) -> bytes:
    started = time.perf_counter()
    first_chunk_at = None
    total_bytes = 0
    status = "ok"
    chunks: list[bytes] = []
    TTS_ACTIVE_REQUESTS.labels(endpoint, mode).inc()

    try:
        pcm = engine.generate(
            text=text,
            mode=mode,
            speaker=speaker,
            instruct=instruct,
            prompt_text=prompt_text,
            prompt_audio=prompt_audio,
            source_audio=source_audio,
            speed=speed,
        )
        chunks.append(pcm)
        total_bytes = len(pcm)
    except Exception as exc:
        status = "error"
        logger.exception("[%s] generate error: %s", req_id, exc)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}") from exc
    finally:
        elapsed = time.perf_counter() - started
        observe_duration_ms(
            TTS_STREAM_COMPLETE_MS,
            (endpoint, mode, lang, status),
            elapsed,
        )
        TTS_REQUESTS_TOTAL.labels(endpoint, mode, lang, status).inc()
        TTS_ACTIVE_REQUESTS.labels(endpoint, mode).dec()

    pcm_bytes = b"".join(chunks)
    output_path = build_audio_output_path(text)
    if pcm_bytes:
        audio_save_queue.put((str(output_path), pcm_bytes))

    sample_rate = current_sample_rate()
    audio_sec = len(pcm_bytes) / 2.0 / sample_rate if pcm_bytes else 0.0
    rtf = elapsed / audio_sec if audio_sec > 0 else 0.0
    if audio_sec > 0:
        TTS_AUDIO_DURATION_SEC.labels(endpoint, mode, lang, status).observe(audio_sec)
        TTS_STREAM_RTF.labels(endpoint, mode, lang, status).observe(rtf)
    logger.info(
        "[%s] fetch_done | bytes=%s | total_ms=%.0f | rtf=%.2f",
        req_id, len(pcm_bytes), elapsed * 1000.0, rtf,
    )
    return pcm_bytes


def validate_config() -> None:
    global COSYVOICE_DEFAULT_SPEED

    if not COSYVOICE_MODEL_DIR:
        raise RuntimeError("COSYVOICE_MODEL_DIR is required")
    if not os.path.isdir(COSYVOICE_MODEL_DIR):
        raise RuntimeError(f"COSYVOICE_MODEL_DIR does not exist: {COSYVOICE_MODEL_DIR}")
    if COSYVOICE_BACKEND not in ("native", "vllm", "trtllm", "trtllm-serve"):
        raise RuntimeError(f"Unsupported COSYVOICE_BACKEND: {COSYVOICE_BACKEND}")
    if COSYVOICE_COMPAT_MODE not in ("sft", "zero_shot", "cross_lingual", "instruct2", "vc"):
        raise RuntimeError(f"Unsupported COSYVOICE_COMPAT_MODE: {COSYVOICE_COMPAT_MODE}")
    raw_speed = os.environ.get("COSYVOICE_DEFAULT_SPEED", COSYVOICE_DEFAULT_SPEED_RAW or "")
    if not raw_speed.strip():
        raise RuntimeError("COSYVOICE_DEFAULT_SPEED is required")
    try:
        COSYVOICE_DEFAULT_SPEED = float(raw_speed)
    except ValueError as exc:
        raise RuntimeError("COSYVOICE_DEFAULT_SPEED must be a float") from exc
    if COSYVOICE_DEFAULT_SPEED <= 0:
        raise RuntimeError("COSYVOICE_DEFAULT_SPEED must be greater than 0")
    if COSYVOICE_BACKEND == "trtllm" and not COSYVOICE_TRT_ENGINE_DIR:
        raise RuntimeError("COSYVOICE_TRT_ENGINE_DIR is required for trtllm backend")
    if COSYVOICE_BACKEND == "trtllm-serve" and not COSYVOICE_TRT_SERVE_URL:
        raise RuntimeError("COSYVOICE_TRT_SERVE_URL is required for trtllm-serve backend")
    GENERATED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def run_warmup() -> None:
    global warmup_completed
    if not TTS_WARMUP_ENABLED:
        warmup_completed = False
        logger.info("TTS warmup is disabled")
        return

    voices = engine.list_voices()
    mode = get_compat_mode()
    speaker = get_request_speaker(voices[0] if mode == "sft" and voices else None)
    text = "Hello, this is a warmup." if mode == "sft" else "Hello."

    logger.info("Running TTS warmup (mode=%s, speaker=%s)...", mode, speaker)
    try:
        if mode == "sft":
            engine.warmup(text=text, mode=mode, speaker=speaker, speed=get_default_speed())
        else:
            engine.warmup(
                text=text,
                mode=mode,
                instruct=get_request_instruct(),
                prompt_text=COSYVOICE_PROMPT_TEXT,
                prompt_audio=load_default_prompt_audio(),
                speed=get_default_speed(),
            )
        warmup_completed = True
    except Exception:
        logger.exception("Warmup failed")
        warmup_completed = False
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, normalizer_pipeline, model_status, model_error

    validate_config()

    logger.info("Initializing TTS engine (backend=%s, model=%s)...", COSYVOICE_BACKEND, COSYVOICE_MODEL_DIR)
    try:
        engine_kwargs: dict = {}
        if COSYVOICE_BACKEND == "native":
            engine_kwargs = {"fp16": COSYVOICE_FP16, "load_trt": COSYVOICE_LOAD_TRT}
        elif COSYVOICE_BACKEND == "vllm":
            engine_kwargs = {"fp16": COSYVOICE_FP16, "gpu_memory_utilization": COSYVOICE_GPU_MEM, "load_trt": COSYVOICE_LOAD_TRT}
        elif COSYVOICE_BACKEND == "trtllm":
            engine_kwargs = {
                "engine_dir": COSYVOICE_TRT_ENGINE_DIR,
                "hf_model_dir": COSYVOICE_HF_MODEL_DIR or None,
                "enable_trt_flow": COSYVOICE_LOAD_TRT,
                "gpu_memory_utilization": COSYVOICE_GPU_MEM,
            }
        elif COSYVOICE_BACKEND == "trtllm-serve":
            engine_kwargs = {
                "serve_url": COSYVOICE_TRT_SERVE_URL,
                "hf_model_dir": COSYVOICE_HF_MODEL_DIR or None,
                "enable_trt_flow": COSYVOICE_LOAD_TRT,
            }

        engine = create_engine(
            backend=COSYVOICE_BACKEND,
            model_dir=COSYVOICE_MODEL_DIR,
            **engine_kwargs,
        )
        model_status = "healthy"
    except Exception as exc:
        model_status = "error"
        model_error = str(exc)
        logger.exception("Failed to initialize TTS engine")
        raise

    normalizer_pipeline = create_normalizer_pipeline(enabled=ENABLE_TEXT_NORMALIZATION)

    ensure_audio_save_worker()
    run_warmup()

    logger.info("CosyVoice3 TTS server ready (backend=%s)", COSYVOICE_BACKEND)
    yield
    logger.info("Server shutdown")


app = FastAPI(
    title="CosyVoice3 TTS Server",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    client = request.client.host if request.client else "unknown"

    body_info = exc.body

    # Convert FormData to a plain dict for logging
    if hasattr(body_info, "multi_items"):
        form_dict = {}
        for key, value in body_info.multi_items():
            if hasattr(value, "filename"):
                form_dict[key] = f"<UploadFile filename={value.filename}>"
            else:
                form_dict[key] = str(value)
        body_info = form_dict

    # If body is missing/empty, try to read raw body or parsed form
    if not body_info:
        try:
            form = await request.form()
            body_info = {}
            for key, value in form.multi_items():
                if hasattr(value, "filename"):
                    body_info[key] = f"<UploadFile filename={value.filename}>"
                else:
                    body_info[key] = str(value)
        except Exception:
            try:
                raw = await request.body()
                body_info = raw.decode("utf-8", errors="replace") if raw else "<empty body>"
            except Exception:
                body_info = "<unable to read body>"

    logger.warning(
        "422 Validation Error | client=%s | path=%s | method=%s | content_type=%s | body=%s | errors=%s",
        client,
        request.url.path,
        request.method,
        request.headers.get("content-type", "unknown"),
        body_info,
        exc.errors(),
    )
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.middleware("http")
async def prometheus_http_middleware(request: Request, call_next):
    started = time.perf_counter()
    path = normalize_http_path(request)
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    except Exception:
        raise
    finally:
        elapsed = time.perf_counter() - started
        HTTP_REQUESTS_TOTAL.labels(request.method, path, status).inc()
        observe_duration_ms(
            HTTP_REQUEST_DURATION_MS,
            (request.method, path),
            elapsed,
        )


@app.get("/")
def root():
    return {
        "message": "CosyVoice3 TTS Server",
        "backend": COSYVOICE_BACKEND,
        "health": "/health",
        "tts_stream": "/tts-stream",
        "openai_compatible": "/v1/audio/speech",
        "openai_wav": "/v1/audio/speech/wav",
        "voices": "/v1/audio/voices",
        "demo": "/demo",
        "sample_rate": current_sample_rate(),
        "api_model": API_MODEL_NAME,
    }


@app.get("/health")
def health():
    health_data = engine.health_check() if engine else {"status": model_status, "error": model_error}
    health_data.update({
        "warmup_enabled": TTS_WARMUP_ENABLED,
        "warmup_completed": warmup_completed,
        "text_normalization_enabled": ENABLE_TEXT_NORMALIZATION,
        "normalizer_available": normalizer_pipeline is not None and any(
            n.supports_language(lang) for lang, normalizers in normalizer_pipeline._normalizers.items() for n in normalizers
        ) if normalizer_pipeline else False,
        "sample_rate": current_sample_rate(),
        "audio_format": "int16",
        "max_text_length": MAX_TEXT_LENGTH,
        "generated_audio_dir": str(GENERATED_AUDIO_DIR),
    })
    if engine:
        health_data["available_speakers"] = engine.list_voices()
    return health_data


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/audio/voices")
def list_voices():
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    voices = engine.list_voices()
    return JSONResponse(content={"voices": voices})


@app.post("/tts-stream")
async def tts_stream(req: TTSStreamRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Text too long (max {MAX_TEXT_LENGTH} chars)")

    req_id = str(uuid.uuid4())
    resolved_lang = resolve_lang(get_request_lang(req))
    mode = get_compat_mode()
    speaker = get_request_speaker(req.speaker)
    instruct = get_request_instruct(req.instruct)

    if normalizer_pipeline:
        text = normalizer_pipeline.normalize(text, resolved_lang)
    text = ensure_cv3_prefix(text)

    prompt_audio_path = load_default_prompt_audio()
    sample_rate = current_sample_rate()

    logger.info(
        "[%s] /tts-stream | mode=%s | request_lang=%s | lang=%s | speaker=%s | chars=%s | sr=%s | barge_in=%s | text=%s",
        req_id,
        mode,
        get_request_lang(req),
        resolved_lang,
        speaker,
        len(text),
        req.sample_rate_hz,
        bool(req.barge_in),
        short_text_for_log(text),
    )

    return StreamingResponse(
        generate_audio_stream(
            text=text,
            mode=mode,
            lang=resolved_lang,
            speaker=speaker,
            instruct=instruct,
            prompt_text=COSYVOICE_PROMPT_TEXT,
            prompt_audio=prompt_audio_path,
            source_audio=None,
            speed=get_default_speed(),
            req_id=req_id,
            endpoint="tts_stream",
        ),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Audio-Format": "int16",
            "X-Channels": "1",
        },
    )


@app.post("/v1/audio/speech")
async def openai_compatible_tts(req: OpenAISpeechRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty input")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Input too long (max {MAX_TEXT_LENGTH} chars)")
    if req.model != API_MODEL_NAME:
        raise HTTPException(status_code=400, detail=f"Only model={API_MODEL_NAME} is supported")
    if req.response_format.lower() != "pcm":
        raise HTTPException(status_code=400, detail="Only response_format=pcm is supported for streaming")
    if not req.stream:
        raise HTTPException(status_code=400, detail="Only stream=true is supported")

    req_id = str(uuid.uuid4())
    lang = resolve_lang(req.lang)
    mode = get_compat_mode()
    speaker = get_request_speaker(req.voice)
    instruct = get_request_instruct(req.instruct)
    prompt_audio = load_default_prompt_audio()

    if normalizer_pipeline:
        text = normalizer_pipeline.normalize(text, lang)
    text = ensure_cv3_prefix(text)
    sample_rate = current_sample_rate()

    logger.info(
        "[%s] /v1/audio/speech | mode=%s | lang=%s | voice=%s | chars=%s",
        req_id, mode, lang, speaker, len(text),
    )

    return StreamingResponse(
        generate_audio_stream(
            text=text,
            mode=mode,
            lang=lang,
            speaker=speaker,
            instruct=instruct,
            prompt_text=COSYVOICE_PROMPT_TEXT,
            prompt_audio=prompt_audio,
            source_audio=None,
            speed=get_default_speed(),
            req_id=req_id,
            endpoint="openai_speech",
        ),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Audio-Format": "int16",
            "X-Channels": "1",
        },
    )


@app.post("/v1/audio/speech/wav")
async def openai_compatible_tts_wav(req: OpenAISpeechRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty input")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Input too long (max {MAX_TEXT_LENGTH} chars)")
    if req.model != API_MODEL_NAME:
        raise HTTPException(status_code=400, detail=f"Only model={API_MODEL_NAME} is supported")

    req_id = str(uuid.uuid4())
    lang = resolve_lang(req.lang)
    mode = get_compat_mode()
    speaker = get_request_speaker(req.voice)
    instruct = get_request_instruct(req.instruct)
    prompt_audio = load_default_prompt_audio()

    if normalizer_pipeline:
        text = normalizer_pipeline.normalize(text, lang)
    text = ensure_cv3_prefix(text)
    sample_rate = current_sample_rate()

    logger.info(
        "[%s] /v1/audio/speech/wav | mode=%s | lang=%s | voice=%s | chars=%s",
        req_id, mode, lang, speaker, len(text),
    )

    pcm_bytes = fetch_audio_bytes(
        text=text,
        mode=mode,
        lang=lang,
        speaker=speaker,
        instruct=instruct,
        prompt_text=COSYVOICE_PROMPT_TEXT,
        prompt_audio=prompt_audio,
        source_audio=None,
        speed=get_default_speed(),
        req_id=req_id,
        endpoint="openai_speech_wav",
    )
    wav_bytes = pcm_bytes_to_wav_bytes(pcm_bytes, sample_rate=sample_rate)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Audio-Format": "wav",
            "X-Channels": "1",
        },
    )


@app.get("/demo")
def demo():
    return HTMLResponse(
        """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>CosyVoice3 TTS Demo</title>
    <style>
      body { font-family: sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
      textarea, input, select, button { width: 100%; box-sizing: border-box; margin-top: 12px; font-size: 16px; }
      textarea { min-height: 100px; padding: 12px; }
      input, select { padding: 8px; }
      button { padding: 12px; cursor: pointer; }
      audio { width: 100%; margin-top: 16px; }
      .row { display: flex; gap: 12px; }
      .row > * { flex: 1; }
    </style>
  </head>
  <body>
    <h1>CosyVoice3 TTS</h1>
    <div class="row">
      <input id="speaker" placeholder="Speaker ID" />
    </div>
    <textarea id="text">Hello, this is a test of the CosyVoice3 text to speech system.</textarea>
    <button id="go">Synthesize</button>
    <audio id="player" controls></audio>
    <script>
      const button = document.getElementById("go");
      const player = document.getElementById("player");
      button.onclick = async () => {
        button.disabled = true;
        button.textContent = "Synthesizing...";
        player.removeAttribute("src");
        try {
          const payload = { text: document.getElementById("text").value };
          const speaker = document.getElementById("speaker").value;
          if (speaker) payload.speaker = speaker;
          const response = await fetch("/tts-stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          });
          if (!response.ok) {
            alert(await response.text());
            return;
          }
          const blob = await response.blob();
          player.src = URL.createObjectURL(blob);
          player.play().catch(() => {});
        } finally {
          button.disabled = false;
          button.textContent = "Synthesize";
        }
      };
    </script>
  </body>
</html>
        """
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=TTS_PORT, log_level="info")
