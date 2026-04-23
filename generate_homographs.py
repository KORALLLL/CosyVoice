import requests
import wave
import os

host = "127.0.0.1"
url = f"http://{host}:8000/tts-stream"
ref_audio = "/workspace/Qwen3-TTS-streaming-custom/tts/wavs/qwen/ref_4.wav"
output_dir = "/workspace/CosyVoice/generated_homographs"
os.makedirs(output_dir, exist_ok=True)

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
    full_text = f"You are a helpful assistant.<|endofprompt|>{text}"
    files = {"prompt_audio": open(ref_audio, "rb")}
    data = {
        "text": full_text,
        "mode": "cross_lingual",
        "lang": "ru",
        "speed": "1.0",
    }
    try:
        r = requests.post(url, data=data, files=files, timeout=300, stream=True)
        if r.status_code == 200:
            raw_path = os.path.join(output_dir, f"homograph_{idx:02d}.raw")
            wav_path = os.path.join(output_dir, f"homograph_{idx:02d}.wav")
            with open(raw_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            # convert raw int16 pcm to wav
            with open(raw_path, "rb") as f:
                pcm = f.read()
            with wave.open(wav_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(24000)
                w.writeframes(pcm)
            duration = len(pcm) / 2 / 24000
            print(f"  -> {wav_path} ({duration:.1f}s)")
        else:
            print(f"  -> FAILED: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"  -> ERROR: {e}")

print("Done!")
