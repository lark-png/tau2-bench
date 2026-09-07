"""Core voice synthesis (TTS) functions with local Kokoro-82M support."""

import os
from dotenv import load_dotenv

from tau2.data_model.audio import AudioData, AudioFormat, AudioEncoding
from tau2.data_model.voice import ElevenLabsTTSConfig
from tau2.utils.retry import tts_retry
from tau2.voice.utils.elevenlabs_utils import tts_elevenlabs
from tau2.data_model.voice_personas import get_persona_name_by_voice_id

load_dotenv()

ProviderConfig = ElevenLabsTTSConfig

# ==================== KOKORO 本地化核心适配区 ====================
_KOKORO_PIPELINE = None

def get_kokoro_pipeline():
    """惰性加载 Kokoro 模型管道，避免在导入时引起不必要的开销和慢启动"""
    global _KOKORO_PIPELINE
    if _KOKORO_PIPELINE is None:
        import torch
        from kokoro import KPipeline
        
        # 检测 CPU / GPU 自动运行
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 'a' 代表美式英语（American English），因为 Kokoro 的主要高质量声音都集中在 a 组
        _KOKORO_PIPELINE = KPipeline(lang_code='a', device=device)
    return _KOKORO_PIPELINE


def tts_kokoro(text: str, config: ProviderConfig) -> AudioData:
    """使用本地轻量离线 Kokoro-82M 模型合成语音"""
    import numpy as np
    from pydub import AudioSegment
    
    # 1. 自动通过原 ElevenLabs 声音 ID 映射至对应的本地角色声音
    # 如果找不到对应的角色，默认回退到美音高自然度女声 'af_bella'
    persona_name = get_persona_name_by_voice_id(config.voice_id)
    
    KOKORO_VOICE_MAP = {
        "matt_delaney": "am_adam",
        "lisa_brenner": "af_bella",
        "mildred_kaplan": "af_nicole",
        "arjun_roy": "am_michael",
        "wei_lin": "af_sarah",
        "mamadou_diallo": "am_puck",
        "priya_patil": "af_heart",
    }
    kokoro_voice = KOKORO_VOICE_MAP.get(persona_name, "af_bella")
    
    # 获取目标采样率（通常框架要求 16000Hz 或 24000Hz）
    target_sample_rate = config.output_audio_format.sample_rate
    
    # 2. 调用本地 Kokoro 进行合成
    pipeline = get_kokoro_pipeline()
    generator = pipeline(text, voice=kokoro_voice, speed=1, split_pattern=r'\n+')
    
    audio_chunks = []
    for _, _, audio in generator:
        if audio is not None and len(audio) > 0:
            audio_chunks.append(audio)
            
    if not audio_chunks:
        raise ValueError(f"Kokoro TTS returned empty audio for text: '{text}'")
        
    # 拼接多个句子音频块
    audio_array = np.concatenate(audio_chunks)
    
    # 3. 将 NumPy float32 [-1.0, 1.0] 的浮点数强制限制并缩放到 16-bit 整数
    audio_array = np.clip(audio_array, -1.0, 1.0)
    pcm_data = (audio_array * 32767).astype(np.int16).tobytes()
    
    # 4. Kokoro 强制生成 24000Hz。若框架目标采样率不等于 24000Hz，使用 pydub 进行高保真重采样
    if target_sample_rate != 24000:
        segment = AudioSegment(
            data=pcm_data,
            sample_width=2,  # 16bit 对应 2 字节
            frame_rate=24000,
            channels=1
        )
        segment = segment.set_frame_rate(target_sample_rate)
        pcm_data = segment.raw_data
        
    # 5. 构建框架期望的 AudioFormat 并包装成 AudioData
    audio_format = AudioFormat(
        encoding=AudioEncoding.PCM_S16LE,
        sample_rate=target_sample_rate,
        channels=1,
    )
    
    return AudioData(data=pcm_data, format=audio_format)
# =================================================================


@tts_retry
def synthesize_voice(
    text: str,
    provider: str,
    provider_config: ProviderConfig,
) -> AudioData:
    """Synthesize voice from text using the specified configuration."""
    # 检查环境变量。如果开启了环境变量 override，强制将 provider 切换成 kokoro
    env_provider = os.getenv("TAU2_TTS_PROVIDER", provider).lower()
    
    if env_provider == "kokoro":
        audio_data = tts_kokoro(text=text, config=provider_config)
    elif env_provider == "elevenlabs":
        audio_data = tts_elevenlabs(text=text, config=provider_config)
    else:
        raise ValueError(f"Unsupported synthesis provider: {env_provider}")

    if not audio_data.format.is_pcm16:
        raise ValueError(
            f"TTS must output PCM_S16LE, got {audio_data.format.encoding}. "
            "Configure the provider to use PCM output format."
        )

    return audio_data