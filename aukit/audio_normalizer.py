#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/11/30
"""
### audio_normalizer
语音正则化，去除音量低的音频段（去除静音），调节音量。
语音正则化方法基于VAD的方法。
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

from scipy.ndimage.morphology import binary_dilation
from pathlib import Path
from typing import Optional, Union
import numpy as np
import librosa
import struct

from .audio_io import Dict2Obj, _sr, _int16_max

try:
    import webrtcvad
except ImportError as e:
    logger.info("ImportError: {}".format(e))


# Default hyperparameters
default_hparams = Dict2Obj(dict(
    int16_max=(2 ** 15) - 1,
    ## Mel-filterbank
    mel_window_length=25,  # In milliseconds
    mel_window_step=10,  # In milliseconds
    mel_n_channels=40,

    ## Audio
    sample_rate=16000,  # sampling_rate
    # Number of spectrogram frames in a partial utterance
    partials_n_frames=160,  # 1600 ms
    # Number of spectrogram frames at inference
    inference_n_frames=80,  # 800 ms

    ## Voice Activation Detection
    # Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
    # This sets the granularity of the VAD. Should not need to be changed.
    vad_window_length=30,  # In milliseconds
    # Number of frames to average together when performing the moving average smoothing.
    # The larger this value, the larger the VAD variations must be to not get smoothed out.
    vad_moving_average_width=8,
    # Maximum number of consecutive silent frames a segment can have.
    vad_max_silence_length=6,

    ## Audio volume normalization
    audio_norm_target_dBFS=-30,
))


def remove_silence(wav, sr=_sr, max_silence_ms=20):
    """
    去除语音中的静音。
    :param wav:
    :param sr:
    :param max_silence_ms: 单位ms
    :return:
    """
    # Compute the voice detection window size
    wav = librosa.resample(wav, orig_sr=sr, target_sr=_sr)

    vad_window_length = 20
    vad_moving_average_width = 10

    samples_per_window = (vad_window_length * _sr) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * _int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2], sample_rate=_sr))
    voice_flags = np.array(voice_flags)

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(max_silence_ms + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    out = wav[audio_mask == True]
    out = librosa.resample(out, orig_sr=_sr, target_sr=sr)
    return out


def tune_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    """
    调节音量大小。
    :param wav:
    :param target_dBFS: 目标音量。
    :param increase_only: 是否只是增加音量。
    :param decrease_only: 是否只是降低音量。
    :return:
    """
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * _int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / _int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray], source_sr: Optional[int] = None, hparams=None):
    """
    预处理语音，去除静音和设置音量。
    :param fpath_or_wav:
    :param source_sr:
    :param hparams:
    :return:
    """
    hparams = hparams or default_hparams
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(fpath_or_wav, sr=None)
    else:
        wav = fpath_or_wav

    # Resample the wav if needed
    if source_sr is not None and source_sr != hparams.sample_rate:
        wav = librosa.resample(wav, source_sr, hparams.sample_rate)

    # Apply the preprocessing: normalize volume and shorten long silences
    wav = tune_volume(wav, hparams.audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav, hparams=hparams)

    return wav


# Smooth the voice detection with a moving average
def moving_average(array, width):
    array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
    ret = np.cumsum(array_padded, dtype=float)
    ret[width:] = ret[width:] - ret[:-width]
    return ret[width - 1:] / width


def trim_long_silences(wav, hparams):
    """去除语音中的静音。(兼容历史版本)"""
    hparams = hparams or default_hparams

    wav = remove_silence(wav,
                         sr=hparams.sample_rate,
                         max_silence_ms=hparams.vad_max_silence_length)
    return wav


if __name__ == "__main__":
    print(__file__)
