from __future__ import annotations

import io
import base64
import logging
import os
import queue
import re
import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import Generator

import numpy as np
import torch

logger = logging.getLogger("cosyvoice3_tts.engine")

SAMPLE_RATE = 24000

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
COSYVOICE_ROOT = os.path.abspath(os.path.join(ROOT_DIR, "../../.."))
MATCHA_ROOT = os.path.join(COSYVOICE_ROOT, "third_party/Matcha-TTS")
TRITON_ROOT = os.path.join(COSYVOICE_ROOT, "runtime/triton_trtllm")

for p in [COSYVOICE_ROOT, MATCHA_ROOT, TRITON_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


def audio_tensor_to_pcm_bytes(tensor: torch.Tensor) -> bytes:
    audio = tensor.squeeze().cpu().numpy()
    pcm = (audio * (2 ** 15)).clip(-32768, 32767).astype(np.int16)
    return pcm.tobytes()


def decode_base64_audio(b64_str: str) -> str:
    import tempfile
    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(data)
        return tmp.name


def pcm_bytes_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def load_trt_tokenizer(hf_model_dir: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            hf_model_dir,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
    except (TypeError, ValueError) as exc:
        logger.info(
            "Tokenizer load with fix_mistral_regex failed for %s (%s); retrying without it",
            hf_model_dir,
            exc,
        )
        return AutoTokenizer.from_pretrained(hf_model_dir, trust_remote_code=True)


def log_trt_tokenizer_metadata(engine_name: str, tokenizer, speech_token_offset: int) -> None:
    special_token_ids = {
        name: getattr(tokenizer, name, None)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    }
    for token in ("<|eos|>", "<|eos1|>"):
        try:
            special_token_ids[token] = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            special_token_ids[token] = None
    chat_template = getattr(tokenizer, "chat_template", None)
    logger.info(
        "%s tokenizer metadata: class=%s chat_template=%r special_token_ids=%s speech_token_offset=%s",
        engine_name,
        tokenizer.__class__.__name__,
        chat_template,
        special_token_ids,
        speech_token_offset,
    )


def read_prompt_audio_bytes(prompt_audio) -> bytes:
    if isinstance(prompt_audio, (bytes, bytearray, memoryview)):
        return bytes(prompt_audio)
    if isinstance(prompt_audio, (str, os.PathLike)):
        with open(os.fspath(prompt_audio), "rb") as f:
            return f.read()
    if hasattr(prompt_audio, "seek"):
        prompt_audio.seek(0)
    if hasattr(prompt_audio, "read"):
        data = prompt_audio.read()
        if isinstance(data, str):
            return data.encode()
        return bytes(data)
    raise TypeError(f"Unsupported prompt_audio type: {type(prompt_audio).__name__}")


class TTSEngine(ABC):
    @abstractmethod
    def stream_generate(
        self,
        text: str,
        mode: str = "zero_shot",
        speaker: str | None = None,
        instruct: str | None = None,
        prompt_text: str | None = None,
        prompt_audio: io.BytesIO | None = None,
        source_audio: io.BytesIO | None = None,
        speed: float = 1.0,
    ) -> Generator[bytes, None, None]:
        ...

    @abstractmethod
    def generate(
        self,
        text: str,
        mode: str = "zero_shot",
        speaker: str | None = None,
        instruct: str | None = None,
        prompt_text: str | None = None,
        prompt_audio: io.BytesIO | None = None,
        source_audio: io.BytesIO | None = None,
        speed: float = 1.0,
    ) -> bytes:
        ...

    @abstractmethod
    def list_voices(self) -> list[str]:
        ...

    @abstractmethod
    def health_check(self) -> dict:
        ...

    def warmup(self, text: str = "Hello, this is a warmup.", mode: str = "sft", **kwargs) -> None:
        logger.info("Running TTS warmup (mode=%s)...", mode)
        started = time.perf_counter()
        try:
            generate_kwargs = {"text": text, "mode": mode, **kwargs}
            pcm = self.generate(**generate_kwargs)
            elapsed = time.perf_counter() - started
            audio_sec = len(pcm) / 2.0 / SAMPLE_RATE
            logger.info("Warmup done: %.2fs audio, %.2fs wall, RTF=%.2f", audio_sec, elapsed, elapsed / audio_sec if audio_sec > 0 else 0)
        except Exception:
            logger.exception("Warmup failed")
            raise


class NativeEngine(TTSEngine):
    def __init__(self, model_dir: str, fp16: bool = True, load_trt: bool = False, trt_concurrent: int = 1):
        from cosyvoice.cli.cosyvoice import CosyVoice3
        self.model_dir = model_dir
        self.fp16 = fp16
        logger.info("Loading CosyVoice3 native model from %s (fp16=%s, trt=%s)", model_dir, fp16, load_trt)
        self.cosyvoice = CosyVoice3(model_dir=model_dir, load_trt=load_trt, fp16=fp16, trt_concurrent=trt_concurrent)
        self._mode_map = {
            "sft": self._infer_sft,
            "zero_shot": self._infer_zero_shot,
            "cross_lingual": self._infer_cross_lingual,
            "instruct2": self._infer_instruct2,
            "vc": self._infer_vc,
        }
        logger.info("NativeEngine ready")

    def _infer_sft(self, text, speaker, speed, stream, **_):
        spk_id = speaker or "default"
        return self.cosyvoice.inference_sft(text, spk_id, stream=stream, speed=speed)

    def _infer_zero_shot(self, text, prompt_text, prompt_audio, speed, stream, **_):
        return self.cosyvoice.inference_zero_shot(text, prompt_text or "", prompt_audio, stream=stream, speed=speed)

    def _infer_cross_lingual(self, text, prompt_audio, speed, stream, **_):
        return self.cosyvoice.inference_cross_lingual(text, prompt_audio, stream=stream, speed=speed)

    def _infer_instruct2(self, text, instruct, prompt_audio, speed, stream, **_):
        return self.cosyvoice.inference_instruct2(text, instruct or "", prompt_audio, stream=stream, speed=speed)

    def _infer_vc(self, source_audio, prompt_audio, speed, stream, **_):
        return self.cosyvoice.inference_vc(source_audio, prompt_audio, stream=stream, speed=speed)

    def _run_inference(self, stream: bool, **kwargs) -> Generator[bytes, None, None] | bytes:
        mode = kwargs.get("mode", "zero_shot")
        if mode not in self._mode_map:
            raise ValueError(f"Unsupported mode: {mode}. Available: {list(self._mode_map.keys())}")
        gen = self._mode_map[mode](stream=stream, **kwargs)
        if stream:
            return (audio_tensor_to_pcm_bytes(chunk["tts_speech"]) for chunk in gen)
        else:
            pcm_parts = []
            for chunk in gen:
                pcm_parts.append(audio_tensor_to_pcm_bytes(chunk["tts_speech"]))
            return b"".join(pcm_parts)

    def stream_generate(self, **kwargs) -> Generator[bytes, None, None]:
        return self._run_inference(stream=True, **kwargs)

    def generate(self, **kwargs) -> bytes:
        return self._run_inference(stream=False, **kwargs)

    def list_voices(self) -> list[str]:
        return self.cosyvoice.list_available_spks()

    def health_check(self) -> dict:
        return {"status": "healthy", "backend": "native", "model_dir": self.model_dir, "fp16": self.fp16}


class VllmEngine(TTSEngine):
    def __init__(self, model_dir: str, fp16: bool = True, gpu_memory_utilization: float = 0.4, load_trt: bool = False, trt_concurrent: int = 1):
        from cosyvoice.cli.cosyvoice import CosyVoice3
        self.model_dir = model_dir
        self.fp16 = fp16
        self.gpu_memory_utilization = gpu_memory_utilization
        self._register_vllm_model()
        self._patch_vllm_config_arch(model_dir)
        os.environ["COSYVOICE_GPU_MEM"] = str(gpu_memory_utilization)
        logger.info("Loading CosyVoice3 with vLLM from %s (gpu_memory_utilization=%s)", model_dir, gpu_memory_utilization)
        self.cosyvoice = CosyVoice3(model_dir=model_dir, load_vllm=True, load_trt=load_trt, fp16=fp16, trt_concurrent=trt_concurrent)
        self._mode_map = {
            "sft": self._infer_sft,
            "zero_shot": self._infer_zero_shot,
            "cross_lingual": self._infer_cross_lingual,
            "instruct2": self._infer_instruct2,
            "vc": self._infer_vc,
        }
        logger.info("VllmEngine ready")

    @staticmethod
    def _register_vllm_model():
        from vllm import ModelRegistry

        try:
            registered = ModelRegistry.is_registered("CosyVoice2ForCausalLM")
        except AttributeError:
            registered = False

        if registered:
            return

        try:
            ModelRegistry.register_model(
                "CosyVoice2ForCausalLM",
                "cosyvoice.vllm.cosyvoice2:CosyVoice2ForCausalLM",
            )
        except ValueError as exc:
            if "already" not in str(exc).lower() and "registered" not in str(exc).lower():
                raise

    @staticmethod
    def _patch_vllm_config_arch(model_dir: str):
        vllm_dir = os.path.join(model_dir, "vllm")
        config_path = os.path.join(vllm_dir, "config.json")
        if not os.path.exists(config_path):
            return
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        archs = config.get("architectures", [])
        if archs == ["CosyVoice2ForCausalLM"]:
            return
        config["architectures"] = ["CosyVoice2ForCausalLM"]
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        logger.info("Patched vLLM config.json architectures to CosyVoice2ForCausalLM")

    def _infer_sft(self, text, speaker, speed, stream, **_):
        spk_id = speaker or "default"
        return self.cosyvoice.inference_sft(text, spk_id, stream=stream, speed=speed)

    def _infer_zero_shot(self, text, prompt_text, prompt_audio, speed, stream, **_):
        return self.cosyvoice.inference_zero_shot(text, prompt_text or "", prompt_audio, stream=stream, speed=speed)

    def _infer_cross_lingual(self, text, prompt_audio, speed, stream, **_):
        return self.cosyvoice.inference_cross_lingual(text, prompt_audio, stream=stream, speed=speed)

    def _infer_instruct2(self, text, instruct, prompt_audio, speed, stream, **_):
        return self.cosyvoice.inference_instruct2(text, instruct or "", prompt_audio, stream=stream, speed=speed)

    def _infer_vc(self, source_audio, prompt_audio, speed, stream, **_):
        return self.cosyvoice.inference_vc(source_audio, prompt_audio, stream=stream, speed=speed)

    def _run_inference(self, stream: bool, **kwargs):
        mode = kwargs.get("mode", "zero_shot")
        if mode not in self._mode_map:
            raise ValueError(f"Unsupported mode: {mode}. Available: {list(self._mode_map.keys())}")
        gen = self._mode_map[mode](stream=stream, **kwargs)
        if stream:
            return (audio_tensor_to_pcm_bytes(chunk["tts_speech"]) for chunk in gen)
        else:
            pcm_parts = []
            for chunk in gen:
                pcm_parts.append(audio_tensor_to_pcm_bytes(chunk["tts_speech"]))
            return b"".join(pcm_parts)

    def stream_generate(self, **kwargs) -> Generator[bytes, None, None]:
        return self._run_inference(stream=True, **kwargs)

    def generate(self, **kwargs) -> bytes:
        return self._run_inference(stream=False, **kwargs)

    def list_voices(self) -> list[str]:
        return self.cosyvoice.list_available_spks()

    def health_check(self) -> dict:
        return {
            "status": "healthy",
            "backend": "vllm",
            "model_dir": self.model_dir,
            "fp16": self.fp16,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }


class _Token2WavRunner:
    def __init__(self, model_dir: str, enable_trt: bool = False, device_id: int = 0, streaming: bool = True):
        from token2wav_cosyvoice3 import CosyVoice3_Token2Wav
        self.model_dir = model_dir
        self.device_id = device_id
        logger.info("Loading Token2Wav from %s (trt=%s)", model_dir, enable_trt)
        self.token2wav = CosyVoice3_Token2Wav(
            model_dir=model_dir, enable_trt=enable_trt, device_id=device_id, streaming=streaming
        )
        logger.info("Token2Wav ready")

    def extract_prompt_features(self, prompt_audio_bytes: bytes, prompt_audio_sr: int = 16000):
        import torchaudio
        try:
            import soundfile as sf
            audio_np, sr = sf.read(io.BytesIO(prompt_audio_bytes), dtype="float32", always_2d=True)
            audio = torch.from_numpy(audio_np).transpose(0, 1)
        except Exception:
            audio, sr = torchaudio.load(io.BytesIO(prompt_audio_bytes), backend="soundfile")
        audio = audio.mean(dim=0)
        if sr != prompt_audio_sr:
            audio = torchaudio.transforms.Resample(sr, prompt_audio_sr)(audio.unsqueeze(0)).squeeze(0)
        prompt_speech_tokens_list = self.token2wav.prompt_audio_tokenization([audio])
        prompt_mels, prompt_mels_lens = self.token2wav.get_prompt_mels([audio], [prompt_audio_sr])
        spk_emb = self.token2wav.get_spk_emb([audio])
        for i in range(len(prompt_speech_tokens_list)):
            token_len = min(int(prompt_mels_lens[i].item() / 2), len(prompt_speech_tokens_list[i]))
            prompt_speech_tokens_list[i] = prompt_speech_tokens_list[i][:token_len]
            prompt_mels_lens[i] = 2 * token_len
        return {
            "prompt_speech_tokens": prompt_speech_tokens_list[0],
            "prompt_mels": prompt_mels,
            "prompt_mels_lens": prompt_mels_lens,
            "spk_emb": spk_emb,
        }

    def stream_token2wav(
        self,
        speech_token_ids: list[int],
        prompt_speech_tokens: list[int],
        prompt_mels: torch.Tensor,
        prompt_mels_lens: torch.Tensor,
        spk_emb: torch.Tensor,
    ) -> Generator[bytes, None, None]:
        prompt_feat = prompt_mels[:1, :prompt_mels_lens[0]]
        embedding = spk_emb[:1]
        total_tokens = len(speech_token_ids)
        token_hop_len = 25
        token_max_hop_len = 4 * token_hop_len
        stream_scale_factor = 2
        token_offset = 0
        current_hop = token_hop_len
        prompt_token_pad = int(
            np.ceil(len(prompt_speech_tokens) / token_hop_len) * token_hop_len
            - len(prompt_speech_tokens)
        )
        hift_cache_mel = None
        speech_offset = 0

        while token_offset < total_tokens:
            this_hop = current_hop + prompt_token_pad if token_offset == 0 else current_hop
            remaining = total_tokens - token_offset
            pre_lookahead_len = self.token2wav.flow.pre_lookahead_len

            if remaining >= this_hop + pre_lookahead_len:
                end_idx = token_offset + this_hop + pre_lookahead_len
                this_token = torch.tensor([speech_token_ids[:end_idx]]).to(self.token2wav.device)
                finalize = False
            else:
                this_token = torch.tensor([speech_token_ids]).to(self.token2wav.device)
                finalize = True

            with torch.cuda.amp.autocast(self.token2wav.fp16):
                mel, _ = self.token2wav.flow.inference(
                    token=this_token,
                    token_len=torch.tensor([this_token.shape[1]]).to(self.token2wav.device),
                    prompt_token=torch.tensor([prompt_speech_tokens]).to(self.token2wav.device),
                    prompt_token_len=torch.tensor([len(prompt_speech_tokens)]).to(self.token2wav.device),
                    prompt_feat=prompt_feat.to(self.token2wav.device),
                    prompt_feat_len=torch.tensor([prompt_feat.shape[1]]).to(self.token2wav.device),
                    embedding=embedding.to(self.token2wav.device),
                    streaming=True,
                    finalize=finalize,
                )

            token_mel_ratio = self.token2wav.flow.token_mel_ratio
            mel = mel[:, :, token_offset * token_mel_ratio:]

            if hift_cache_mel is not None:
                mel = torch.concat([hift_cache_mel, mel], dim=2)
            hift_cache_mel = mel

            tts_speech, _ = self.token2wav.hift.inference(speech_feat=mel, finalize=finalize)
            tts_speech = tts_speech[:, speech_offset:]
            speech_offset += tts_speech.shape[1]

            pcm = audio_tensor_to_pcm_bytes(tts_speech)
            yield pcm

            token_offset += this_hop
            if not finalize:
                current_hop = min(token_max_hop_len, current_hop * stream_scale_factor)
            else:
                break

    def full_token2wav(
        self,
        speech_token_ids: list[int],
        prompt_speech_tokens: list[int],
        prompt_mels: torch.Tensor,
        prompt_mels_lens: torch.Tensor,
        spk_emb: torch.Tensor,
    ) -> bytes:
        pcm_parts = list(self.stream_token2wav(
            speech_token_ids, prompt_speech_tokens, prompt_mels, prompt_mels_lens, spk_emb
        ))
        return b"".join(pcm_parts)


class TrtLlmEngine(TTSEngine):
    _COSYVOICE3_PREFIX = "You are a helpful assistant.<|endofprompt|>"
    _COSYVOICE3_PROMPT_MARKER = "<|endofprompt|>"

    def __init__(self, model_dir: str, engine_dir: str, hf_model_dir: str | None = None,
                 enable_trt_flow: bool = False, device_id: int = 0,
                 gpu_memory_utilization: float = 0.6, max_batch_size: int = 1):
        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunnerCpp

        self.model_dir = model_dir
        self.engine_dir = engine_dir
        self.device_id = device_id

        if hf_model_dir is None:
            hf_model_dir = os.path.join(model_dir, "hf_merged")

        logger.info("Loading TRT-LLM engine from %s", engine_dir)
        self.tokenizer = load_trt_tokenizer(hf_model_dir)

        runtime_rank = 0
        runner_kwargs = dict(
            engine_dir=engine_dir,
            rank=runtime_rank,
            max_output_len=2048,
            enable_context_fmha_fp32_acc=False,
            max_batch_size=max_batch_size,
            max_input_len=512,
            kv_cache_free_gpu_memory_fraction=gpu_memory_utilization,
            cuda_graph_mode=False,
            gather_generation_logits=False,
        )
        self.runner = ModelRunnerCpp.from_dir(**runner_kwargs)
        logger.info("TRT-LLM engine ready")

        self.t2w = _Token2WavRunner(model_dir=model_dir, enable_trt=enable_trt_flow, device_id=device_id, streaming=True)

        metadata_path = os.path.join(hf_model_dir, "cosyvoice3_metadata.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        self.speech_token_offset = self.metadata.get("speech_token_offset", 0)
        self.base_speech_token_size = self.metadata.get("base_speech_token_size", 6561)
        self.eos_speech_idx = self.base_speech_token_size + 1
        log_trt_tokenizer_metadata("TrtLlmEngine", self.tokenizer, self.speech_token_offset)

        logger.info("TrtLlmEngine ready")

    @classmethod
    def _ensure_single_cv3_prefix(cls, content: str) -> str:
        if content.startswith(cls._COSYVOICE3_PREFIX):
            return content
        if cls._COSYVOICE3_PREFIX in content:
            return cls._COSYVOICE3_PREFIX + content.replace(cls._COSYVOICE3_PREFIX, "", 1)
        if cls._COSYVOICE3_PROMPT_MARKER in content:
            return content
        return cls._COSYVOICE3_PREFIX + content

    def _build_chat(self, text: str, mode: str, prompt_text: str | None = None,
                    instruct: str | None = None,
                    prompt_speech_token_ids: list[int] | None = None) -> list[dict]:
        if mode == "sft":
            content = self._ensure_single_cv3_prefix(text)
            chat = [{"role": "user", "content": content}]
        elif mode == "zero_shot":
            content = self._ensure_single_cv3_prefix(f"{prompt_text or ''}{text}")
            assistant_content = ""
            if prompt_speech_token_ids:
                assistant_content = "".join(f"<|s_{tid}|>" for tid in prompt_speech_token_ids)
            chat = [{"role": "user", "content": content}]
            if assistant_content:
                chat.append({"role": "assistant", "content": assistant_content})
        elif mode == "cross_lingual":
            content = self._ensure_single_cv3_prefix(text)
            assistant_content = ""
            if prompt_speech_token_ids:
                assistant_content = "".join(f"<|s_{tid}|>" for tid in prompt_speech_token_ids)
            chat = [{"role": "user", "content": content}]
            if assistant_content:
                chat.append({"role": "assistant", "content": assistant_content})
        elif mode == "instruct2":
            content = self._ensure_single_cv3_prefix(f"{prompt_text or instruct or ''}{text}")
            chat = [{"role": "user", "content": content}]
        else:
            raise ValueError(f"Unsupported mode for TRT-LLM: {mode}")
        return chat

    def _extract_speech_ids(self, output_ids: list[int]) -> list[int]:
        speech_ids = []
        offset = self.speech_token_offset
        for tid in output_ids:
            if offset <= tid < offset + self.base_speech_token_size + 200:
                local_id = tid - offset
                if local_id == self.eos_speech_idx:
                    break
                if local_id < self.base_speech_token_size:
                    speech_ids.append(local_id)
        return speech_ids

    def _run_llm(self, input_ids: torch.Tensor) -> list[int]:
        input_lengths = [input_ids.size(0)]
        end_id = self.tokenizer.convert_tokens_to_ids("<|eos1|>") if "<|eos1|>" in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id
        outputs = self.runner.generate(
            batch_input_ids=[input_ids],
            max_new_tokens=2048,
            end_id=end_id,
            pad_id=end_id,
            temperature=0.8,
            top_k=25,
            top_p=0.95,
            repetition_penalty=1.1,
            num_return_sequences=1,
            streaming=False,
            output_sequence_lengths=True,
            output_generation_logits=False,
            return_dict=True,
            return_all_generated_tokens=False,
        )
        torch.cuda.synchronize()
        output_ids_tensor = outputs["output_ids"]
        sequence_lengths = outputs["sequence_lengths"]
        output_begin = input_lengths[0]
        output_end = sequence_lengths[0][0]
        return output_ids_tensor[0][0][output_begin:output_end].tolist()

    def _run_pipeline(self, stream: bool, **kwargs) -> Generator[bytes, None, None] | bytes:
        text = kwargs["text"]
        mode = kwargs.get("mode", "zero_shot")
        instruct = kwargs.get("instruct")
        prompt_text = kwargs.get("prompt_text")
        prompt_audio = kwargs.get("prompt_audio")

        prompt_features = None
        prompt_speech_token_ids = None
        if prompt_audio is not None and mode in ("zero_shot", "cross_lingual", "instruct2"):
            prompt_audio_bytes = read_prompt_audio_bytes(prompt_audio)
            prompt_features = self.t2w.extract_prompt_features(prompt_audio_bytes)
            prompt_speech_token_ids = prompt_features["prompt_speech_tokens"]

        chat = self._build_chat(text, mode, prompt_text, instruct, prompt_speech_token_ids)
        continue_final = len(chat) > 1
        input_ids = self.tokenizer.apply_chat_template(
            chat, tokenize=True, return_tensors="pt", continue_final_message=continue_final
        ).squeeze(0)

        output_ids = self._run_llm(input_ids)
        speech_ids = self._extract_speech_ids(output_ids)

        if not speech_ids:
            logger.warning("No speech tokens generated for text: %s", text[:100])
            return iter([]) if stream else b""

        if prompt_features is None:
            prompt_features = self._get_default_prompt_features()

        if stream:
            return self.t2w.stream_token2wav(
                speech_ids,
                prompt_features["prompt_speech_tokens"],
                prompt_features["prompt_mels"],
                prompt_features["prompt_mels_lens"],
                prompt_features["spk_emb"],
            )
        else:
            return self.t2w.full_token2wav(
                speech_ids,
                prompt_features["prompt_speech_tokens"],
                prompt_features["prompt_mels"],
                prompt_features["prompt_mels_lens"],
                prompt_features["spk_emb"],
            )

    def _get_default_prompt_features(self):
        if not hasattr(self, "_default_prompt_features"):
            default_audio_path = os.path.join(self.model_dir, "assets", "default_prompt.wav")
            if os.path.exists(default_audio_path):
                with open(default_audio_path, "rb") as f:
                    prompt_audio_bytes = f.read()
                features = self.t2w.extract_prompt_features(prompt_audio_bytes)
            else:
                features = {
                    "prompt_speech_tokens": [],
                    "prompt_mels": torch.zeros(1, 0, 80),
                    "prompt_mels_lens": torch.tensor([0]),
                    "spk_emb": torch.zeros(1, 192),
                }
            self._default_prompt_features = features
        return self._default_prompt_features

    def stream_generate(self, **kwargs) -> Generator[bytes, None, None]:
        return self._run_pipeline(stream=True, **kwargs)

    def generate(self, **kwargs) -> bytes:
        return self._run_pipeline(stream=False, **kwargs)

    def list_voices(self) -> list[str]:
        spk2info_path = os.path.join(self.model_dir, "spk2info.pt")
        if os.path.exists(spk2info_path):
            return list(torch.load(spk2info_path, map_location="cpu", weights_only=True).keys())
        return []

    def health_check(self) -> dict:
        return {"status": "healthy", "backend": "trtllm", "model_dir": self.model_dir, "engine_dir": self.engine_dir}


class TrtLlmServeEngine(TTSEngine):
    _SPEECH_TOKEN_RE = re.compile(r"<\|s_(\d+)\|>")
    _EOS_TOKEN_RE = re.compile(r"<\|eos\d*\|>")
    _COSYVOICE3_PREFIX = "You are a helpful assistant.<|endofprompt|>"
    _COSYVOICE3_PROMPT_MARKER = "<|endofprompt|>"
    _MAX_NEW_SPEECH_TOKENS = 2048
    _MIN_TOKEN_TEXT_RATIO = 2
    _MAX_TOKEN_TEXT_RATIO = 20

    def __init__(self, model_dir: str, serve_url: str, model_name: str = "trt_engines_bfloat16",
                 enable_trt_flow: bool = False, device_id: int = 0,
                 hf_model_dir: str | None = None, timeout: float = 300.0):
        import httpx

        self.model_dir = model_dir
        self.serve_url = serve_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

        if hf_model_dir is None:
            hf_model_dir = os.path.join(model_dir, "hf_merged")

        self.tokenizer = load_trt_tokenizer(hf_model_dir)
        self.http_client = httpx.Client(timeout=timeout)

        self.t2w = _Token2WavRunner(model_dir=model_dir, enable_trt=enable_trt_flow, device_id=device_id, streaming=True)

        metadata_path = os.path.join(hf_model_dir, "cosyvoice3_metadata.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        self.speech_token_offset = self.metadata.get("speech_token_offset", 0)
        self.base_speech_token_size = self.metadata.get("base_speech_token_size", 6561)
        self.eos_speech_idx = self.base_speech_token_size + 1
        log_trt_tokenizer_metadata("TrtLlmServeEngine", self.tokenizer, self.speech_token_offset)

        logger.info("TrtLlmServeEngine ready (url=%s)", serve_url)

    @classmethod
    def _ensure_single_cv3_prefix(cls, content: str) -> str:
        if content.startswith(cls._COSYVOICE3_PREFIX):
            return content
        if cls._COSYVOICE3_PREFIX in content:
            return cls._COSYVOICE3_PREFIX + content.replace(cls._COSYVOICE3_PREFIX, "", 1)
        if cls._COSYVOICE3_PROMPT_MARKER in content:
            return content
        return cls._COSYVOICE3_PREFIX + content

    def _build_chat(self, text: str, mode: str, prompt_text: str | None = None,
                    instruct: str | None = None,
                    prompt_speech_token_ids: list[int] | None = None) -> list[dict]:
        if mode == "sft":
            content = self._ensure_single_cv3_prefix(text)
            chat = [{"role": "user", "content": content}]
        elif mode == "zero_shot":
            content = self._ensure_single_cv3_prefix(f"{prompt_text or ''}{text}")
            assistant_content = ""
            if prompt_speech_token_ids:
                assistant_content = "".join(f"<|s_{tid}|>" for tid in prompt_speech_token_ids)
            chat = [{"role": "user", "content": content}]
            if assistant_content:
                chat.append({"role": "assistant", "content": assistant_content})
        elif mode == "cross_lingual":
            content = self._ensure_single_cv3_prefix(text)
            chat = [{"role": "user", "content": content}]
        elif mode == "instruct2":
            content = self._ensure_single_cv3_prefix(f"{instruct or prompt_text or ''}{text}")
            chat = [{"role": "user", "content": content}]
        else:
            raise ValueError(f"Unsupported mode for trtllm-serve: {mode}")
        return chat

    def _extract_speech_ids_from_str(self, token_strs: list[str]) -> list[int]:
        speech_ids = []
        for s in token_strs:
            s = self._EOS_TOKEN_RE.split(s, maxsplit=1)[0]
            for match in self._SPEECH_TOKEN_RE.finditer(s):
                num = int(match.group(1))
                if num == self.eos_speech_idx:
                    return speech_ids
                if num < self.base_speech_token_size:
                    speech_ids.append(num)
        return speech_ids

    @classmethod
    def _strip_generation_prefix(cls, text: str) -> str:
        if text.startswith(cls._COSYVOICE3_PREFIX):
            return text[len(cls._COSYVOICE3_PREFIX):]
        marker_index = text.rfind(cls._COSYVOICE3_PROMPT_MARKER)
        if marker_index >= 0:
            return text[marker_index + len(cls._COSYVOICE3_PROMPT_MARKER):]
        return text

    def _count_text_tokens(self, text: str) -> int:
        text = text or ""
        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            try:
                return len(encode(text, add_special_tokens=False))
            except TypeError:
                return len(encode(text))
        tokenize = getattr(self.tokenizer, "tokenize", None)
        if callable(tokenize):
            return len(tokenize(text))
        return max(1, len(text.split()))

    def _speech_token_bounds(self, text: str, mode: str, prompt_text: str | None, instruct: str | None) -> tuple[int, int]:
        target_text = self._strip_generation_prefix(text)
        target_token_count = max(1, self._count_text_tokens(target_text))
        min_tokens = max(1, target_token_count * self._MIN_TOKEN_TEXT_RATIO)
        max_tokens = min(self._MAX_NEW_SPEECH_TOKENS, max(min_tokens, target_token_count * self._MAX_TOKEN_TEXT_RATIO))
        return min_tokens, max_tokens

    @staticmethod
    def _min_pcm_bytes_for_tokens(min_tokens: int) -> int:
        min_audio_sec = max(0.05, min_tokens / 25.0 * 0.5)
        return int(min_audio_sec * SAMPLE_RATE * 2)

    def _validate_pcm_length(self, pcm: bytes, min_tokens: int) -> None:
        min_bytes = self._min_pcm_bytes_for_tokens(min_tokens)
        if len(pcm) < min_bytes:
            raise RuntimeError(
                f"TRT token2wav produced too little PCM: {len(pcm)} bytes < {min_bytes} bytes"
            )

    def _run_llm_serve(self, chat: list[dict], min_tokens: int, max_tokens: int) -> list[int]:
        payload = {
            "model": self.model_name,
            "messages": chat,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 25,
            "repetition_penalty": 1.1,
            "stop": ["<|eos1|>", "<|eos|>"],
            "stream": False,
        }

        response = self.http_client.post(
            f"{self.serve_url}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
        generated_content = result["choices"][0]["message"]["content"]
        speech_ids = self._extract_speech_ids_from_str([generated_content])
        logger.info(
            "TRT-LLM serve generated %s speech tokens (min=%s, max=%s, content_chars=%s, preview=%r)",
            len(speech_ids),
            min_tokens,
            max_tokens,
            len(generated_content),
            generated_content[:160],
        )
        return speech_ids

    def _run_pipeline(self, stream: bool, **kwargs):
        text = kwargs["text"]
        mode = kwargs.get("mode", "zero_shot")
        prompt_text = kwargs.get("prompt_text")
        instruct = kwargs.get("instruct")
        prompt_audio = kwargs.get("prompt_audio")

        prompt_features = None
        prompt_speech_token_ids = None
        if prompt_audio is not None and mode in ("zero_shot", "cross_lingual", "instruct2"):
            prompt_audio_bytes = read_prompt_audio_bytes(prompt_audio)
            prompt_features = self.t2w.extract_prompt_features(prompt_audio_bytes)
            prompt_speech_token_ids = prompt_features["prompt_speech_tokens"]

        chat = self._build_chat(text, mode, prompt_text, instruct, prompt_speech_token_ids)
        min_tokens, max_tokens = self._speech_token_bounds(text, mode, prompt_text, instruct)
        speech_ids = self._run_llm_serve(chat, min_tokens=min_tokens, max_tokens=max_tokens)

        if not speech_ids:
            raise RuntimeError(f"TRT-LLM serve generated no speech tokens for text: {text[:100]}")
        if len(speech_ids) < min_tokens:
            raise RuntimeError(
                f"TRT-LLM serve generated too few speech tokens: {len(speech_ids)} < {min_tokens}"
            )

        if prompt_features is None:
            prompt_features = self._get_default_prompt_features()

        if stream:
            return self.t2w.stream_token2wav(
                speech_ids,
                prompt_features["prompt_speech_tokens"],
                prompt_features["prompt_mels"],
                prompt_features["prompt_mels_lens"],
                prompt_features["spk_emb"],
            )
        else:
            pcm = self.t2w.full_token2wav(
                speech_ids,
                prompt_features["prompt_speech_tokens"],
                prompt_features["prompt_mels"],
                prompt_features["prompt_mels_lens"],
                prompt_features["spk_emb"],
            )
            self._validate_pcm_length(pcm, min_tokens)
            logger.info(
                "TRT-LLM serve token2wav produced %s bytes (%.2fs)",
                len(pcm),
                len(pcm) / 2.0 / SAMPLE_RATE,
            )
            return pcm

    def _get_default_prompt_features(self):
        if not hasattr(self, "_default_prompt_features"):
            self._default_prompt_features = {
                "prompt_speech_tokens": [],
                "prompt_mels": torch.zeros(1, 0, 80),
                "prompt_mels_lens": torch.tensor([0]),
                "spk_emb": torch.zeros(1, 192),
            }
        return self._default_prompt_features

    def stream_generate(self, **kwargs) -> Generator[bytes, None, None]:
        return self._run_pipeline(stream=True, **kwargs)

    def generate(self, **kwargs) -> bytes:
        return self._run_pipeline(stream=False, **kwargs)

    def list_voices(self) -> list[str]:
        spk2info_path = os.path.join(self.model_dir, "spk2info.pt")
        if os.path.exists(spk2info_path):
            return list(torch.load(spk2info_path, map_location="cpu", weights_only=True).keys())
        return []

    def health_check(self) -> dict:
        try:
            resp = self.http_client.get(f"{self.serve_url}/v1/models", timeout=5.0)
            return {"status": "healthy" if resp.status_code == 200 else "error", "backend": "trtllm-serve", "serve_url": self.serve_url}
        except Exception as e:
            return {"status": "error", "backend": "trtllm-serve", "serve_url": self.serve_url, "error": str(e)}


def create_engine(backend: str, model_dir: str, **kwargs) -> TTSEngine:
    backend = backend.lower().strip()
    if backend == "native":
        return NativeEngine(model_dir=model_dir, **kwargs)
    elif backend == "vllm":
        return VllmEngine(model_dir=model_dir, **kwargs)
    elif backend == "trtllm":
        return TrtLlmEngine(model_dir=model_dir, **kwargs)
    elif backend == "trtllm-serve":
        return TrtLlmServeEngine(model_dir=model_dir, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose from: native, vllm, trtllm, trtllm-serve")
