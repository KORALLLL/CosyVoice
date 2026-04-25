import os
import sys

sys.path.insert(0, '/workspace/CosyVoice')
sys.path.insert(0, '/workspace/CosyVoice/third_party/Matcha-TTS')
sys.path.insert(0, '/workspace/CosyVoice/runtime/python/fastapi')

os.environ.setdefault('COSYVOICE_MODEL_DIR', '/workspace/CosyVoice/pretrained_models/FunAudioLLM/Fun-CosyVoice3-0___5B-2512')
os.environ.setdefault('COSYVOICE_BACKEND', 'vllm')
os.environ.setdefault('COSYVOICE_FP16', 'true')
os.environ.setdefault('COSYVOICE_LOAD_TRT', 'false')
os.environ.setdefault('COSYVOICE_DEFAULT_SPEED', '1.0')
os.environ.setdefault('VLLM_PLUGINS', 'cosyvoice')
os.environ.setdefault('TTS_PORT', '8003')
os.environ.setdefault('TTS_HOST', '0.0.0.0')
os.environ.setdefault('TTS_WARMUP_ENABLED', 'false')
os.environ.setdefault('GENERATED_AUDIO_DIR', '/workspace/CosyVoice/generated')

import uvicorn
from server_cosyvoice3 import app

host = os.environ.get('TTS_HOST', '0.0.0.0')
port = int(os.environ.get('TTS_PORT', '8003'))
uvicorn.run(app, host=host, port=port, log_level='info')
