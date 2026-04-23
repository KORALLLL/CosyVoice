import os
import sys

sys.path.insert(0, '/workspace/CosyVoice')
sys.path.insert(0, '/workspace/CosyVoice/third_party/Matcha-TTS')
sys.path.insert(0, '/workspace/CosyVoice/runtime/python/fastapi')

os.environ.setdefault('COSYVOICE_MODEL_DIR', '/workspace/CosyVoice/pretrained_models/FunAudioLLM/Fun-CosyVoice3-0___5B-2512')
os.environ.setdefault('COSYVOICE_BACKEND', 'native')
os.environ.setdefault('COSYVOICE_FP16', 'true')
os.environ.setdefault('COSYVOICE_LOAD_TRT', 'false')
os.environ.setdefault('TTS_PORT', '8000')
os.environ.setdefault('TTS_WARMUP_ENABLED', 'false')
os.environ.setdefault('GENERATED_AUDIO_DIR', '/workspace/CosyVoice/generated')

import uvicorn
from server_cosyvoice3 import app

host = '.'.join(['0', '0', '0', '0'])
uvicorn.run(app, host=host, port=8000, log_level='info')
