import json
import math
import os
import sys
import wave
from array import array
from pathlib import Path

import requests

host = os.environ.get("TTS_HOST", "127.0.0.1")
port = os.environ.get("TTS_PORT", "8003")
url = os.environ.get("TTS_URL", f"http://{host}:{port}/tts-stream")
output_dir = os.environ.get("OUTPUT_DIR", "/workspace/CosyVoice/generated_homographs")
min_duration_sec = float(os.environ.get("MIN_DURATION_SEC", "1.0"))
manifest_path = os.environ.get(
    "MANIFEST_PATH",
    os.path.join(
        os.path.dirname(output_dir),
        f"{os.path.basename(os.path.normpath(output_dir))}.manifest.jsonl",
    ),
)
enable_gigaam_asr = os.environ.get("ENABLE_GIGAAM_ASR", "").lower() == "true"
gigaam_root = os.environ.get("GIGAAM_ROOT", "/workspace/GigaAM")
silence_dbfs = float(os.environ.get("SILENCE_DBFS", "-50.0"))
compat_mode = os.environ.get("COSYVOICE_COMPAT_MODE", "cross_lingual")
configured_backend = os.environ.get("COSYVOICE_BACKEND")
prompt_audio_path = os.environ.get("COSYVOICE_DEFAULT_REF_AUDIO") or os.environ.get(
    "DEFAULT_PROMPT_AUDIO"
)
health_url = os.environ.get("TTS_HEALTH_URL", f"http://{host}:{port}/health")


def read_validated_pcm(response):
    sample_rate = int(response.headers.get("X-Sample-Rate", "24000"))
    audio_format = response.headers.get("X-Audio-Format", "").lower()
    channels = response.headers.get("X-Channels", "1")
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
    chunks = []

    if audio_format and audio_format != "int16":
        raise ValueError(f"unexpected X-Audio-Format={audio_format!r}")
    if channels != "1":
        raise ValueError(f"unexpected X-Channels={channels!r}")
    if content_type and content_type not in ("application/octet-stream", "audio/pcm"):
        raise ValueError(f"unexpected Content-Type={content_type!r}")

    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            chunks.append(chunk)

    pcm = b"".join(chunks)
    if len(pcm) % 2:
        raise ValueError(f"PCM byte count is odd: {len(pcm)}")

    duration = len(pcm) / 2 / sample_rate
    if duration < min_duration_sec:
        raise ValueError(f"audio too short: {duration:.2f}s < {min_duration_sec:.2f}s")

    return pcm, sample_rate, duration


def save_pcm_as_wav(pcm, sample_rate, wav_path):
    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)


def _pcm_to_float_samples(pcm):
    try:
        import numpy as np

        return np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    except Exception:
        samples = array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        return [sample / 32768.0 for sample in samples]


def _count_leading(values, predicate):
    count = 0
    for value in values:
        if not predicate(value):
            break
        count += 1
    return count


def compute_audio_metrics(pcm, sample_rate):
    samples = _pcm_to_float_samples(pcm)
    total_samples = len(samples)
    if total_samples == 0:
        return {
            "duration_sec": 0.0,
            "silence_ratio": 1.0,
            "leading_silence_sec": 0.0,
            "trailing_silence_sec": 0.0,
            "rms_dbfs": None,
            "peak_dbfs": None,
        }

    silence_threshold = 10 ** (silence_dbfs / 20.0)

    try:
        import numpy as np

        abs_samples = np.abs(samples)
        silent = abs_samples <= silence_threshold
        peak = float(abs_samples.max())
        rms = float(np.sqrt(np.mean(samples * samples)))
        silence_count = int(silent.sum())
        leading_silence = 0
        for is_silent in silent:
            if not bool(is_silent):
                break
            leading_silence += 1
        trailing_silence = 0
        for is_silent in silent[::-1]:
            if not bool(is_silent):
                break
            trailing_silence += 1
    except Exception:
        abs_samples = [abs(sample) for sample in samples]
        silent = [sample <= silence_threshold for sample in abs_samples]
        peak = max(abs_samples)
        rms = math.sqrt(sum(sample * sample for sample in samples) / total_samples)
        silence_count = sum(1 for is_silent in silent if is_silent)
        leading_silence = _count_leading(silent, bool)
        trailing_silence = _count_leading(reversed(silent), bool)

    def to_dbfs(value):
        if value <= 0.0:
            return None
        return 20.0 * math.log10(value)

    return {
        "duration_sec": total_samples / sample_rate,
        "silence_ratio": silence_count / total_samples,
        "leading_silence_sec": leading_silence / sample_rate,
        "trailing_silence_sec": trailing_silence / sample_rate,
        "rms_dbfs": to_dbfs(rms),
        "peak_dbfs": to_dbfs(peak),
    }


def load_gigaam_model():
    root = Path(gigaam_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import gigaam

    kwargs = {}
    download_root = os.environ.get("GIGAAM_DOWNLOAD_ROOT")
    if download_root:
        kwargs["download_root"] = download_root
    return gigaam.load_model("v3_ctc", **kwargs)


def transcribe_with_gigaam(model, wav_path):
    result = model.transcribe(wav_path)
    return getattr(result, "text", result)


def get_server_info():
    try:
        response = requests.get(health_url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return {}
    return {}


def response_token_count(response):
    for header in ("X-Speech-Token-Count", "X-Generated-Speech-Tokens"):
        value = response.headers.get(header)
        if value:
            try:
                return int(value)
            except ValueError:
                return None
    return None


def empty_manifest_record(idx, text, wav_path, server_info=None):
    server_info = server_info or {}
    return {
        "index": idx,
        "text": text,
        "output_path": wav_path,
        "status": "error",
        "error": None,
        "url": url,
        "backend": configured_backend or server_info.get("backend"),
        "mode": compat_mode,
        "prompt_audio_path": prompt_audio_path,
        "token_count": None,
        "sample_rate": None,
        "duration_sec": None,
        "silence_ratio": None,
        "leading_silence_sec": None,
        "trailing_silence_sec": None,
        "rms_dbfs": None,
        "peak_dbfs": None,
        "asr_transcript": None,
    }


def write_manifest_record(manifest_file, record):
    manifest_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_file.flush()


sentences = [
    "Я регулярно плачу за ваши услуги без задержек, но последний счёт вызвал у меня такое недоумение, что я готов плакать от бессилия, ведь сумма не соответствует нашему договору.",
    "Наш специалист приедет и починит замок вашей входной двери уже завтра, а в выходные вы можете отдохнуть и посетить замок, который мы рекомендуем в нашем туристическом пакете.",
    "Ваш брак с нашим банком продолжается уже десять лет, и мы ценим это партнёрство, но к сожалению, последняя партия карт оказалась бракованной, поэтому мы заменим их бесплатно.",
    "Для приготовления блюда по нашему рецепту вам нужна мука тонкого помола, а не мука от бесконечных звонков в колл-центр, где вас держат в ожидании более часа.",
    "Мы отправили вам ключ для входа в личный кабинет на имеил, приложили ключ для регулировки оборудования, и дали ключ к пониманию вашей проблемы в инструкции.",
    "Возьмите ручку и запишите номер заявки, а если у вас сломалась ручка на двери балкона, сообщите нам, и мы отправим мастера вместе с курьером.",
    "Проверьте каждый лист договора перед подписанием, ведь даже один упавший лист осенью меняет пейзаж, и одна ошибка в документе меняет условия сделки.",
    "Мы ценим, что вы остаетесь с нами, ведь каждый клиент — свет в окне нашего офиса, и мы хотим, чтобы свет вашей лояльности никогда не угас.",
    "Дорога к решению вашего вопроса оказалась долгой, но теперь, когда проблема устранена, ваша поддержка нам дорога, как никогда раньше.",
    "Если батарейка в вашем устройстве села, вставьте новую батарейку, следуя инструкции, а если ваш садовый участок зарос, вам понадобится коса для уборки травы.",
]


def main():
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    asr_model = None
    server_info = get_server_info()

    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        for idx, text in enumerate(sentences, 1):
            print(f"Generating {idx}/{len(sentences)}...")
            wav_path = os.path.join(output_dir, f"homograph_{idx:02d}.wav")
            record = empty_manifest_record(idx, text, wav_path, server_info)
            payload = {
                "text": text,
                "lang": "ru",
            }
            try:
                r = requests.post(url, json=payload, timeout=300, stream=True)
                if r.status_code == 200:
                    pcm, sample_rate, duration = read_validated_pcm(r)
                    save_pcm_as_wav(pcm, sample_rate, wav_path)
                    metrics = compute_audio_metrics(pcm, sample_rate)
                    record.update(metrics)
                    record.update(
                        {
                            "status": "ok",
                            "sample_rate": sample_rate,
                            "duration_sec": duration,
                            "token_count": response_token_count(r),
                            "error": None,
                        }
                    )
                    if asr_model is not None:
                        record["asr_transcript"] = transcribe_with_gigaam(
                            asr_model, wav_path
                        )
                    elif enable_gigaam_asr:
                        asr_model = load_gigaam_model()
                        record["asr_transcript"] = transcribe_with_gigaam(
                            asr_model, wav_path
                        )
                    print(f"  -> {wav_path} ({duration:.1f}s)")
                else:
                    record["error"] = f"HTTP {r.status_code}: {r.text[:200]}"
                    print(f"  -> FAILED: {record['error']}")
            except Exception as e:
                record["status"] = "error"
                record["error"] = str(e)
                print(f"  -> ERROR: {e}")
            finally:
                write_manifest_record(manifest_file, record)

    print(f"Manifest: {manifest_path}")
    print("Done!")


if __name__ == "__main__":
    main()
