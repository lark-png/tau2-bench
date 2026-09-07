import soundfile as sf
from kokoro import KPipeline
import torch

# 1. 初始化本地合成管道，'a' 代表美式英语 (American English)
# 如果服务器有 GPU，它会自动使用 GPU 进行极速推理
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"正在初始化 Kokoro 管道，使用设备: {device}")
pipeline = KPipeline(lang_code='a', device=device) 

TEXT = "Hello, I am calling about my flight booking. It seems there is a delay."
# 'af_bella' 是一个非常自然的高质量本地女声（Bella）
VOICE = 'af_bella' 

print("正在本地合成语音...")
# 进行合成
generator = pipeline(TEXT, voice=VOICE, speed=1, split_pattern=r'\n+')

for i, (graphemes, phonemes, audio) in enumerate(generator):
    # audio 是一个 numpy 数组，可以直接保存为 wav 格式，采样率 24000Hz
    sf.write("test_offline_tts.wav", audio, 24000)
    print(f"本地合成成功！音频已保存至: test_offline_tts.wav")