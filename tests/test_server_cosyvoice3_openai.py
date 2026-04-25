from __future__ import annotations

import asyncio
import importlib.util
import io
import os
import sys
import tempfile
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
ENGINE_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "runtime"
    / "python"
    / "fastapi"
    / "tts_engine.py"
)
LONG_TEST_PCM = b"\x01\x00" * 2400


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


def load_tts_engine_module():
    fake_transformers = types.ModuleType("transformers")
    fake_httpx = types.ModuleType("httpx")
    fake_torch = types.ModuleType("torch")
    fake_numpy = types.ModuleType("numpy")

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    class FakeHttpClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_httpx.Client = FakeHttpClient
    fake_torch.Tensor = object
    fake_torch.zeros = lambda *args, **kwargs: {"zeros": args, "kwargs": kwargs}
    fake_torch.tensor = lambda value, *args, **kwargs: value
    fake_torch.load = lambda *args, **kwargs: {}
    fake_numpy.int16 = int
    fake_numpy.ceil = lambda value: value

    spec = importlib.util.spec_from_file_location(
        "tts_engine_test_module",
        ENGINE_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    patched_modules = {
        "transformers": fake_transformers,
        "httpx": fake_httpx,
        "torch": fake_torch,
        "numpy": fake_numpy,
    }
    with mock.patch.dict(sys.modules, patched_modules):
        spec.loader.exec_module(module)
    return module


class FakeToken2WavRunner:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.extracted_prompt_audio: bytes | None = None
        self.stream_calls: list[dict] = []
        self.full_calls: list[dict] = []
        self.full_pcm = LONG_TEST_PCM

    def extract_prompt_features(self, prompt_audio_bytes: bytes):
        self.extracted_prompt_audio = prompt_audio_bytes
        return {
            "prompt_speech_tokens": [41, 42],
            "prompt_mels": "prompt-mels",
            "prompt_mels_lens": "prompt-mels-lens",
            "spk_emb": "speaker-embedding",
        }

    def stream_token2wav(
        self,
        speech_token_ids,
        prompt_speech_tokens,
        prompt_mels,
        prompt_mels_lens,
        spk_emb,
    ):
        self.stream_calls.append(
            {
                "speech_token_ids": speech_token_ids,
                "prompt_speech_tokens": prompt_speech_tokens,
                "prompt_mels": prompt_mels,
                "prompt_mels_lens": prompt_mels_lens,
                "spk_emb": spk_emb,
            }
        )
        yield b"stream-pcm"

    def full_token2wav(
        self,
        speech_token_ids,
        prompt_speech_tokens,
        prompt_mels,
        prompt_mels_lens,
        spk_emb,
    ):
        self.full_calls.append(
            {
                "speech_token_ids": speech_token_ids,
                "prompt_speech_tokens": prompt_speech_tokens,
                "prompt_mels": prompt_mels,
                "prompt_mels_lens": prompt_mels_lens,
                "spk_emb": spk_emb,
            }
        )
        return self.full_pcm


class FakeServeResponse:
    def __init__(self, status_code: int = 200, content: str = "<|s_7|><|s_8|>"):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


class FakeServeClient:
    def __init__(self, content: str = "<|s_7|><|s_8|>", health_status: int = 200):
        self.content = content
        self.health_status = health_status
        self.posts: list[dict] = []
        self.gets: list[dict] = []

    def post(self, url, json):
        self.posts.append({"url": url, "json": json})
        return FakeServeResponse(content=self.content)

    def get(self, url, timeout):
        self.gets.append({"url": url, "timeout": timeout})
        return FakeServeResponse(status_code=self.health_status)


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


class TrtLlmServeEngineContractTests(unittest.TestCase):
    def make_engine(self, content: str = "<|s_7|><|s_8|>", health_status: int = 200):
        module = load_tts_engine_module()
        fake_transformers = types.ModuleType("transformers")
        fake_httpx = types.ModuleType("httpx")

        class FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        fake_httpx.Client = lambda *args, **kwargs: FakeServeClient()
        with mock.patch.object(module, "_Token2WavRunner", FakeToken2WavRunner):
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers, "httpx": fake_httpx}):
                engine = module.TrtLlmServeEngine(
                    model_dir="/models/cosyvoice3",
                    serve_url="http://trtllm.test:8000/",
                    model_name="configured-trt-model",
                    hf_model_dir="/models/cosyvoice3/hf",
                    timeout=12.5,
                )
        engine.http_client = FakeServeClient(content=content, health_status=health_status)
        return engine

    def test_prompt_audio_path_from_env_contract_is_accepted(self) -> None:
        engine = self.make_engine()
        with tempfile.NamedTemporaryFile(suffix=".wav") as prompt_audio:
            prompt_audio.write(b"prompt-audio-from-env")
            prompt_audio.flush()

            pcm = engine.generate(
                text="hello",
                mode="cross_lingual",
                prompt_audio=prompt_audio.name,
            )

        self.assertEqual(pcm, LONG_TEST_PCM)
        self.assertEqual(engine.t2w.extracted_prompt_audio, b"prompt-audio-from-env")

    def test_upstream_payload_uses_existing_config_not_request_params(self) -> None:
        engine = self.make_engine()

        pcm = engine.generate(
            text="hello",
            mode="cross_lingual",
            prompt_audio=io.BytesIO(b"prompt-audio"),
            model="request-model-must-not-leak",
            temperature=0.01,
            top_p=0.1,
            speed=3.0,
        )

        self.assertEqual(pcm, LONG_TEST_PCM)
        request = engine.http_client.posts[0]
        self.assertEqual(request["url"], "http://trtllm.test:8000/v1/chat/completions")
        self.assertEqual(request["json"]["model"], "configured-trt-model")
        self.assertEqual(request["json"]["temperature"], 0.8)
        self.assertEqual(request["json"]["top_p"], 0.95)
        self.assertEqual(request["json"]["min_tokens"], 2)
        self.assertEqual(request["json"]["max_tokens"], 20)
        self.assertFalse(request["json"]["stream"])
        self.assertNotIn("continue_final_message", request["json"])
        self.assertNotIn("speed", request["json"])
        self.assertNotIn("voice", request["json"])
        self.assertNotIn("lang", request["json"])

    def test_cross_lingual_does_not_leak_prompt_text_or_speech_tokens_to_llm(self) -> None:
        engine = self.make_engine()

        pcm = engine.generate(
            text=f"{engine._COSYVOICE3_PREFIX}hello",
            mode="cross_lingual",
            prompt_text="reference prompt. ",
            prompt_audio=io.BytesIO(b"prompt-audio"),
        )

        self.assertEqual(pcm, LONG_TEST_PCM)
        messages = engine.http_client.posts[0]["json"]["messages"]
        self.assertEqual(len(messages), 1)
        content = messages[0]["content"]
        self.assertEqual(content.count(engine._COSYVOICE3_PREFIX), 1)
        self.assertTrue(content.startswith(engine._COSYVOICE3_PREFIX))
        self.assertNotIn("reference prompt.", content)
        self.assertEqual(content, f"{engine._COSYVOICE3_PREFIX}hello")

    def test_zero_shot_still_conditions_llm_on_prompt_text_and_speech_tokens(self) -> None:
        engine = self.make_engine()

        pcm = engine.generate(
            text="hello",
            mode="zero_shot",
            prompt_text="reference prompt. ",
            prompt_audio=io.BytesIO(b"prompt-audio"),
        )

        self.assertEqual(pcm, LONG_TEST_PCM)
        messages = engine.http_client.posts[0]["json"]["messages"]
        self.assertEqual(messages[0]["content"], f"{engine._COSYVOICE3_PREFIX}reference prompt. hello")
        self.assertEqual(messages[1], {"role": "assistant", "content": "<|s_41|><|s_42|>"})

    def test_parses_contiguous_and_separated_speech_tokens(self) -> None:
        cases = {
            "contiguous": "<|s_7|><|s_8|><|s_9|>",
            "separated": "<|s_7|> <|s_8|>\n<|s_9|>",
            "stops_at_eos_marker": "<|s_7|><|s_8|><|eos1|><|s_99|> garbage <|s_100|>",
        }
        for name, content in cases.items():
            with self.subTest(name=name):
                engine = self.make_engine(content=content)

                pcm = engine.generate(
                    text="hello",
                    mode="cross_lingual",
                    prompt_audio=io.BytesIO(b"prompt-audio"),
                )

                self.assertEqual(pcm, LONG_TEST_PCM)
                expected_ids = [7, 8] if name == "stops_at_eos_marker" else [7, 8, 9]
                self.assertEqual(engine.t2w.full_calls[0]["speech_token_ids"], expected_ids)

    def test_raises_when_no_speech_tokens_are_returned(self) -> None:
        engine = self.make_engine(content="plain text without speech tokens")

        with self.assertRaisesRegex(RuntimeError, "generated no speech tokens"):
            engine.generate(
                text="hello",
                mode="cross_lingual",
                prompt_audio=io.BytesIO(b"prompt-audio"),
            )
        self.assertEqual(engine.t2w.full_calls, [])

    def test_raises_when_trt_returns_fewer_tokens_than_requested_minimum(self) -> None:
        engine = self.make_engine(content="<|s_7|><|s_8|>")

        with self.assertRaisesRegex(RuntimeError, "too few speech tokens"):
            engine.generate(
                text="hello world",
                mode="cross_lingual",
                prompt_audio=io.BytesIO(b"prompt-audio"),
            )
        self.assertEqual(engine.t2w.full_calls, [])

        request = engine.http_client.posts[0]["json"]
        self.assertEqual(request["min_tokens"], 4)
        self.assertEqual(request["max_tokens"], 40)

    def test_raises_when_token2wav_returns_implausibly_short_pcm(self) -> None:
        engine = self.make_engine(content="<|s_7|><|s_8|>")
        engine.t2w.full_pcm = b"\x00\x00"

        with self.assertRaisesRegex(RuntimeError, "too little PCM"):
            engine.generate(
                text="hello",
                mode="cross_lingual",
                prompt_audio=io.BytesIO(b"prompt-audio"),
            )

    def test_health_reports_trtllm_serve_status_from_models_endpoint(self) -> None:
        healthy = self.make_engine(health_status=200)
        unhealthy = self.make_engine(health_status=503)

        self.assertEqual(
            healthy.health_check(),
            {
                "status": "healthy",
                "backend": "trtllm-serve",
                "serve_url": "http://trtllm.test:8000",
            },
        )
        self.assertEqual(healthy.http_client.gets[0]["url"], "http://trtllm.test:8000/v1/models")
        self.assertEqual(healthy.http_client.gets[0]["timeout"], 5.0)
        self.assertEqual(unhealthy.health_check()["status"], "error")


class TrtLlmEngineContractTests(unittest.TestCase):
    def test_prompt_audio_path_bytes_and_file_like_are_accepted(self) -> None:
        module = load_tts_engine_module()

        with tempfile.NamedTemporaryFile(suffix=".wav") as prompt_audio:
            prompt_audio.write(b"path-audio")
            prompt_audio.flush()

            self.assertEqual(module.read_prompt_audio_bytes(prompt_audio.name), b"path-audio")

        self.assertEqual(module.read_prompt_audio_bytes(b"bytes-audio"), b"bytes-audio")
        self.assertEqual(module.read_prompt_audio_bytes(io.BytesIO(b"file-like-audio")), b"file-like-audio")

    def test_tokenizer_loader_tries_mistral_regex_fix_then_falls_back(self) -> None:
        module = load_tts_engine_module()
        fake_transformers = types.ModuleType("transformers")
        calls: list[dict] = []

        class FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls, *_args, **kwargs):
                calls.append(kwargs)
                if kwargs.get("fix_mistral_regex"):
                    raise TypeError("unsupported kwarg")
                return cls()

        fake_transformers.AutoTokenizer = FakeAutoTokenizer

        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            tokenizer = module.load_trt_tokenizer("/models/cosyvoice3/hf")

        self.assertIsInstance(tokenizer, FakeAutoTokenizer)
        self.assertEqual(calls[0]["fix_mistral_regex"], True)
        self.assertEqual(calls[0]["trust_remote_code"], True)
        self.assertEqual(calls[1], {"trust_remote_code": True})

    def test_text_with_cosyvoice3_prefix_is_not_prefixed_again(self) -> None:
        module = load_tts_engine_module()
        engine = object.__new__(module.TrtLlmEngine)

        chat = engine._build_chat(
            text=f"{engine._COSYVOICE3_PREFIX}hello",
            mode="cross_lingual",
        )

        content = chat[0]["content"]
        self.assertEqual(content.count(engine._COSYVOICE3_PREFIX), 1)
        self.assertEqual(content, f"{engine._COSYVOICE3_PREFIX}hello")


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

    def test_lifespan_passes_trtllm_serve_model_name_override(self) -> None:
        module = load_module()
        calls = []

        def fake_create_engine(*args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return FakeEngine()

        async def enter_lifespan():
            cm = module.lifespan(None)
            await cm.__aenter__()
            await cm.__aexit__(None, None, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            module.create_engine = fake_create_engine
            module.COSYVOICE_MODEL_DIR = str(Path(__file__).resolve().parents[1])
            module.COSYVOICE_BACKEND = "trtllm-serve"
            module.COSYVOICE_TRT_SERVE_URL = "http://trtllm.test:8000"
            module.COSYVOICE_TRT_SERVE_MODEL_NAME = "serve-env-model"
            module.COSYVOICE_DEFAULT_SPEED_RAW = "1.0"
            module.TTS_WARMUP_ENABLED = False
            module.GENERATED_AUDIO_DIR = Path(tmpdir)
            module.ensure_audio_save_worker = lambda: None

            asyncio.run(enter_lifespan())

        self.assertEqual(calls[0]["kwargs"]["backend"], "trtllm-serve")
        self.assertEqual(calls[0]["kwargs"]["serve_url"], "http://trtllm.test:8000")
        self.assertEqual(calls[0]["kwargs"]["model_name"], "serve-env-model")

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
