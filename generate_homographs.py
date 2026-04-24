import os
import wave

import requests

host = os.environ.get("TTS_HOST", "127.0.0.1")
port = os.environ.get("TTS_PORT", "8000")
url = os.environ.get("TTS_URL", f"http://{host}:{port}/tts-stream")
output_dir = os.environ.get("OUTPUT_DIR", "/workspace/CosyVoice/generated_homographs")
os.makedirs(output_dir, exist_ok=True)


def save_pcm_stream_as_wav(response, wav_path):
    sample_rate = int(response.headers.get("X-Sample-Rate", "24000"))
    total_bytes = 0

    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                wav_file.writeframes(chunk)
                total_bytes += len(chunk)

    return total_bytes / 2 / sample_rate

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

for idx, text in enumerate(sentences, 1):
    print(f"Generating {idx}/10...")
    payload = {
        "text": text,
        "lang": "ru",
    }
    try:
        r = requests.post(url, json=payload, timeout=300, stream=True)
        if r.status_code == 200:
            wav_path = os.path.join(output_dir, f"homograph_{idx:02d}.wav")
            duration = save_pcm_stream_as_wav(r, wav_path)
            print(f"  -> {wav_path} ({duration:.1f}s)")
        else:
            print(f"  -> FAILED: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"  -> ERROR: {e}")

print("Done!")
