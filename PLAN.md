# CosyVoice Qwen-Compatible TTS API

## Goal

Implement the active CosyVoice3 FastAPI service so a Qwen3 TTS client can switch only the service URL and path base, while CosyVoice runs through the in-process vLLM backend.

## Completed Checklist

- [x] Explore `/workspace/Qwen3-TTS-streaming-custom` endpoint and schema contract.
- [x] Explore `/workspace/CosyVoice` active FastAPI and engine paths.
- [x] Convert `runtime/python/fastapi/server_cosyvoice3.py` to Qwen-compatible public schemas.
- [x] Add Qwen-compatible endpoints: `/`, `/health`, `/metrics`, `/v1/audio/voices`, `/tts-stream`, `/v1/audio/speech`, `/v1/audio/speech/wav`.
- [x] Keep CosyVoice-only generation parameters out of public request models.
- [x] Use internal `cross_lingual` mode by default.
- [x] Require `COSYVOICE_DEFAULT_SPEED` at startup and pass it internally.
- [x] Copy Qwen reference audio to `asset/qwen_ref_4.wav`.
- [x] Apply CosyVoice3 prefix handling internally.
- [x] Use the CosyVoice model sample rate dynamically; current local config resolves to 24000 Hz.
- [x] Default launchers to `COSYVOICE_BACKEND=vllm`.
- [x] Install the vLLM stack in `.venv` with `uv`.
- [x] Wire `COSYVOICE_GPU_MEM` to vLLM `gpu_memory_utilization`.
- [x] Ensure `CosyVoice2ForCausalLM` registration works in vLLM parent and worker processes.
- [x] Generate vLLM config with the correct architecture before engine startup.
- [x] Fix CosyVoice vLLM LM-head bias handling for the exported local checkpoint.
- [x] Add lightweight compatibility tests for schemas, validation, headers, sample rate, and WAV wrapping.
- [x] Run a separate validation-agent pass after worker integration.
- [x] Real-smoke `COSYVOICE_BACKEND=vllm` with `/health`.
- [x] Real-smoke `/v1/audio/speech/wav` and verify mono int16 WAV at 24000 Hz.

## Implementation Notes

- Public request models are intentionally Qwen-compatible; CosyVoice defaults are configured by env/internal constants.
- The default prompt reference file is `asset/qwen_ref_4.wav`, copied byte-for-byte from `/workspace/Qwen3-TTS-streaming-custom/tts/wavs/qwen/ref_4.wav`.
- `COSYVOICE_GPU_MEM` is passed into vLLM as `gpu_memory_utilization`; the smoke run used `0.4`.
- vLLM custom model registration is available through the local editable plugin in `vllm_cosyvoice_plugin/`.
- `asset/qwen_ref_4.wav` and model export files are ignored by the repo `.gitignore`, but the files exist locally.

## Validation

- [x] `python -m py_compile runtime/python/fastapi/server_cosyvoice3.py runtime/python/fastapi/tts_engine.py cosyvoice/cli/model.py cosyvoice/vllm/cosyvoice2.py tests/test_server_cosyvoice3_openai.py`
- [x] `bash -n start_server.sh run_server.sh runtime/python/fastapi/start_server.sh`
- [x] `COSYVOICE_MODEL_DIR=... COSYVOICE_DEFAULT_SPEED=1.0 .venv/bin/python -m unittest discover -s tests -v`
- [x] `curl http://127.0.0.1:8010/health`
- [x] `POST /v1/audio/speech/wav` generated `/tmp/cosyvoice_smoke.wav`: RIFF WAV, mono, 16-bit, 24000 Hz.

## Follow-Up

- [ ] Decide whether to vendor or force-add ignored runtime assets such as `asset/qwen_ref_4.wav` for version control.
- [ ] Consider making generation non-blocking for health probes during long synchronous WAV requests.
- [ ] Implement the backend proxy wrapper after the CosyVoice vLLM service is accepted.
