from __future__ import annotations

import asyncio
import importlib.util
import io
import os
import sys
import types
import unittest
import wave
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "runtime"
    / "python"
    / "fastapi"
    / "server_cosyvoice3.py"
)


def _pcm_bytes_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def load_module():
    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi_exceptions = types.ModuleType("fastapi.exceptions")
    fake_fastapi_middleware = types.ModuleType("fastapi.middleware")
    fake_fastapi_cors = types.ModuleType("fastapi.middleware.cors")
    fake_fastapi_requests = types.ModuleType("fastapi.requests")
    fake_fastapi_responses = types.ModuleType("fastapi.responses")
    fake_pydantic = types.ModuleType("pydantic")
    fake_prometheus = types.ModuleType("prometheus_client")
    fake_tts_engine = types.ModuleType("tts_engine")
    fake_normalizer = types.ModuleType("normalizer")

    class FakeFastAPI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def add_middleware(self, *args, **kwargs):
            return None

        def exception_handler(self, *_args, **_kwargs):
            def decorator(func):
                return func

            return decorator

        def middleware(self, *_args, **_kwargs):
            def decorator(func):
                return func

            return decorator

        def get(self, *_args, **_kwargs):
            def decorator(func):
                return func

            return decorator

        def post(self, *_args, **_kwargs):
            def decorator(func):
                return func

            return decorator

    class FakeHTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FakeRequestValidationError(Exception):
        pass

    class FakeUploadFile:
        filename = "fake.wav"
        file = io.BytesIO()

    class FakeRequest:
        scope = {}
        url = types.SimpleNamespace(path="/")
        client = types.SimpleNamespace(host="testclient")
        method = "POST"
        headers = {}

    class FakeResponse:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.content = kwargs.get("content", args[0] if args else b"")
            self.media_type = kwargs.get("media_type")
            self.headers = kwargs.get("headers", {})
            self.status_code = kwargs.get("status_code", 200)

    class FakeStreamingResponse(FakeResponse):
        def __init__(self, body_iterator, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.body_iterator = body_iterator

    class FakeBaseModel:
        def __init__(self, **kwargs):
            fields: dict[str, object] = {}
            for cls in reversed(type(self).__mro__):
                fields.update(getattr(cls, "__annotations__", {}))
            for key in kwargs:
                if key not in fields:
                    raise TypeError(f"unexpected field: {key}")
            for key in fields:
                if key in kwargs:
                    setattr(self, key, kwargs[key])
                elif hasattr(type(self), key):
                    setattr(self, key, getattr(type(self), key))
                else:
                    setattr(self, key, None)

    class FakeMetric:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            return None

        def dec(self, *args, **kwargs):
            return None

        def observe(self, *args, **kwargs):
            return None

    class FakeTTSEngine:
        pass

    class FakeNormalizerPipeline:
        _normalizers = {}

    fake_fastapi.FastAPI = FakeFastAPI
    fake_fastapi.HTTPException = FakeHTTPException
    fake_fastapi.UploadFile = FakeUploadFile
    fake_fastapi.File = lambda *args, **kwargs: None
    fake_fastapi.Form = lambda *args, **kwargs: None
    fake_fastapi_exceptions.RequestValidationError = FakeRequestValidationError
    fake_fastapi_cors.CORSMiddleware = object
    fake_fastapi_requests.Request = FakeRequest
    fake_fastapi_responses.HTMLResponse = FakeResponse
    fake_fastapi_responses.JSONResponse = FakeResponse
    fake_fastapi_responses.Response = FakeResponse
    fake_fastapi_responses.StreamingResponse = FakeStreamingResponse
    fake_pydantic.BaseModel = FakeBaseModel
    fake_prometheus.CONTENT_TYPE_LATEST = "text/plain"
    fake_prometheus.Counter = lambda *args, **kwargs: FakeMetric()
    fake_prometheus.Gauge = lambda *args, **kwargs: FakeMetric()
    fake_prometheus.Histogram = lambda *args, **kwargs: FakeMetric()
    fake_prometheus.generate_latest = lambda: b""
    fake_tts_engine.TTSEngine = FakeTTSEngine
    fake_tts_engine.create_engine = lambda *args, **kwargs: FakeTTSEngine()
    fake_tts_engine.decode_base64_audio = lambda _b64: io.BytesIO(b"decoded-audio")
    fake_tts_engine.pcm_bytes_to_wav_bytes = _pcm_bytes_to_wav_bytes
    fake_normalizer.NormalizerPipeline = FakeNormalizerPipeline
    fake_normalizer.create_normalizer_pipeline = lambda enabled=True: None

    patched_modules = {
        "fastapi": fake_fastapi,
        "fastapi.exceptions": fake_fastapi_exceptions,
        "fastapi.middleware": fake_fastapi_middleware,
        "fastapi.middleware.cors": fake_fastapi_cors,
        "fastapi.requests": fake_fastapi_requests,
        "fastapi.responses": fake_fastapi_responses,
        "pydantic": fake_pydantic,
        "prometheus_client": fake_prometheus,
        "tts_engine": fake_tts_engine,
        "normalizer": fake_normalizer,
    }
    spec = importlib.util.spec_from_file_location(
        "server_cosyvoice3_test_module",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    with mock.patch.dict(sys.modules, patched_modules):
        spec.loader.exec_module(module)
    return module


class FakeEngine:
    def __init__(self, pcm: bytes = b"\x01\x00\x02\x00"):
        self.pcm = pcm
        self.stream_calls: list[dict] = []
        self.generate_calls: list[dict] = []

    def stream_generate(self, **kwargs):
        self.stream_calls.append(kwargs)
        midpoint = max(2, len(self.pcm) // 2)
        yield self.pcm[:midpoint]
        yield self.pcm[midpoint:]

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return self.pcm

    def list_voices(self):
        return ["test-voice"]

    def health_check(self):
        return {"status": "healthy"}


class OpenAICompatibilityTests(unittest.TestCase):
    def make_request(self, module, **overrides):
        payload = {
            "input": "hello from tests",
            "model": module.API_MODEL_NAME,
            "voice": None,
            "response_format": "pcm",
            "stream": True,
            "instruct": None,
            "lang": module.DEFAULT_LANG,
        }
        payload.update(overrides)
        return types.SimpleNamespace(**payload)

    def assert_http_error(self, module, coro, status_code: int, detail_fragment: str) -> None:
        with self.assertRaises(module.HTTPException) as ctx:
            asyncio.run(coro)
        self.assertEqual(ctx.exception.status_code, status_code)
        self.assertIn(detail_fragment, ctx.exception.detail)

    def test_openai_speech_request_schema_fields_and_defaults(self) -> None:
        module = load_module()

        request = module.OpenAISpeechRequest(input="hello")

        self.assertEqual(
            list(module.OpenAISpeechRequest.__annotations__.keys()),
            ["input", "model", "voice", "response_format", "stream", "lang", "instruct"],
        )
        self.assertEqual(request.input, "hello")
        self.assertEqual(request.model, module.API_MODEL_NAME)
        self.assertIsNone(request.voice)
        self.assertEqual(request.response_format, "pcm")
        self.assertTrue(request.stream)
        self.assertEqual(request.lang, module.DEFAULT_LANG)
        self.assertIsNone(request.instruct)

    def test_tts_stream_request_schema_matches_qwen_fields(self) -> None:
        module = load_module()

        self.assertEqual(
            list(module.TTSStreamRequest.__annotations__.keys()),
            [
                "text",
                "lang",
                "language",
                "speaker",
                "instruct",
                "sample_rate_hz",
                "codec",
                "bitrate",
                "bandwidth_hz_est",
                "snr_db",
                "silence_ratio",
                "rms_dbfs",
                "packet_loss_pct",
                "jitter_ms",
                "barge_in",
            ],
        )
        with self.assertRaises(TypeError):
            module.TTSStreamRequest(text="hello", mode="zero_shot")

    def test_validate_config_requires_default_speed(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            module = load_module()
            module.COSYVOICE_MODEL_DIR = str(Path(__file__).resolve().parents[1])
            module.GENERATED_AUDIO_DIR = Path("/tmp/cosyvoice-test-generated")

            with self.assertRaises(RuntimeError) as ctx:
                module.validate_config()

        self.assertIn("COSYVOICE_DEFAULT_SPEED is required", str(ctx.exception))

    def test_wrong_model_is_rejected(self) -> None:
        module = load_module()
        module.engine = FakeEngine()
        module.normalizer_pipeline = None

        self.assert_http_error(
            module,
            module.openai_compatible_tts(
                self.make_request(module, model="not-cosyvoice3"),
            ),
            400,
            "model",
        )

    def test_stream_false_is_rejected(self) -> None:
        module = load_module()
        module.engine = FakeEngine()
        module.normalizer_pipeline = None

        self.assert_http_error(
            module,
            module.openai_compatible_tts(self.make_request(module, stream=False)),
            400,
            "Only stream=true is supported",
        )

    def test_non_pcm_streaming_response_format_is_rejected(self) -> None:
        module = load_module()
        module.engine = FakeEngine()
        module.normalizer_pipeline = None

        self.assert_http_error(
            module,
            module.openai_compatible_tts(
                self.make_request(module, response_format="wav"),
            ),
            400,
            "Only response_format=pcm is supported",
        )

    def test_streaming_response_has_pcm_headers_and_sample_rate(self) -> None:
        module = load_module()
        engine = FakeEngine()
        module.engine = engine
        module.normalizer_pipeline = None

        response = asyncio.run(
            module.openai_compatible_tts(self.make_request(module, lang="ru")),
        )

        self.assertEqual(response.media_type, "application/octet-stream")
        self.assertEqual(response.headers["X-Sample-Rate"], str(module.SAMPLE_RATE))
        self.assertEqual(response.headers["X-Audio-Format"], "int16")
        self.assertEqual(response.headers["X-Channels"], "1")
        self.assertEqual(b"".join(response.body_iterator), engine.pcm)
        self.assertEqual(engine.stream_calls[0]["mode"], "cross_lingual")
        self.assertEqual(engine.stream_calls[0]["speed"], 1.0)
        self.assertEqual(
            engine.stream_calls[0]["prompt_audio"],
            module.DEFAULT_PROMPT_AUDIO,
        )

    def test_streaming_headers_use_dynamic_engine_sample_rate(self) -> None:
        module = load_module()
        engine = FakeEngine()
        engine.sample_rate = 16000
        module.engine = engine
        module.normalizer_pipeline = None

        response = asyncio.run(
            module.openai_compatible_tts(self.make_request(module, lang="ru")),
        )

        self.assertEqual(response.headers["X-Sample-Rate"], "16000")

    def test_wav_endpoint_wraps_generated_pcm_in_riff_wav(self) -> None:
        module = load_module()
        engine = FakeEngine(pcm=b"\x01\x00\x02\x00\x03\x00\x04\x00")
        module.engine = engine
        module.normalizer_pipeline = None

        response = asyncio.run(
            module.openai_compatible_tts_wav(
                self.make_request(module, response_format="wav"),
            ),
        )

        self.assertEqual(response.media_type, "audio/wav")
        self.assertEqual(response.headers["X-Sample-Rate"], str(module.SAMPLE_RATE))
        self.assertEqual(response.headers["X-Audio-Format"], "wav")
        self.assertTrue(response.content.startswith(b"RIFF"))

        with wave.open(io.BytesIO(response.content), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), module.SAMPLE_RATE)
            self.assertEqual(wf.readframes(wf.getnframes()), engine.pcm)


if __name__ == "__main__":
    unittest.main()
