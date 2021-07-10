#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/12/1
"""
### audio_io
语音IO，语音保存、读取，支持wav和mp3格式，语音形式转换（np.array,bytes,io.BytesIO），支持【.】操作符的字典。
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

from scipy.io import wavfile
from pathlib import Path
import numpy as np
import librosa
import io
import json
from dotmap import DotMap

_sr = 16000
_int16_max = 2 ** 15 - 1


class Dict2Obj(DotMap):
    """
    修正DotMap的get方法生成DotMap对象的bug。
    Dict2Obj的get方法和dict的get功能相同。
    """

    def __getitem__(self, k):
        if k not in self._map:
            return None
        else:
            return self._map[k]

    def parse(self, json_string):
        if json_string.strip():
            _hp = json.loads(json_string)
            for k, v in _hp.items():
                self[k] = v
        return self


def load_wav(path, sr=None, with_sr=False):
    """
    导入语音信号。支持wav和mp3格式。
    :param path: 文件路径。
    :param sr: 采样率，None: 自动识别采样率。
    :param with_sr: 是否返回采样率。
    :return: np.ndarray
    """
    return load_wav_librosa(path, sr=sr, with_sr=with_sr)


def save_wav(wav, path, sr=_sr):
    save_wav_wavfile(wav, path=path, sr=sr)


def load_wav_librosa(path, sr=_sr, with_sr=False):
    wav, sr = librosa.load(path, sr=sr)
    return (wav, sr) if with_sr else wav


def load_wav_wavfile(path, sr=None, with_sr=False):
    sr, wav = wavfile.read(path)
    wav = wav / max(np.max(np.abs(wav)), 0.01)
    return (wav, sr) if with_sr else wav


def save_wav_librosa(wav, path, sr=_sr):
    librosa.output.write_wav(path, wav, sr=sr)


def save_wav_wavfile(wav, path, sr=_sr, volume=1.):
    out = wav * _int16_max * volume / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, out.astype(np.int16))


def anything2bytesio(src, sr=_sr, volume=1.):
    if isinstance(src, (str, Path)):
        src = load_wav(src, sr=sr)
    if isinstance(src, (list, np.ndarray, np.matrix)):
        out_io = io.BytesIO()
        save_wav_wavfile(src, out_io, sr=sr, volume=volume)
    elif isinstance(src, bytes):
        out_io = io.BytesIO(src)
    elif isinstance(src, io.BytesIO):
        out_io = src
    else:
        raise TypeError
    return out_io


def anything2wav(src, sr=_sr, volume=1.):
    if isinstance(src, (list, np.ndarray, np.matrix)):
        return np.array(src)
    else:
        bysio = anything2bytesio(src, sr=sr, volume=volume)
        return load_wav_wavfile(bysio, sr=sr)


def anything2bytes(src, sr=_sr, volume=1.):
    if isinstance(src, bytes):
        return src
    else:
        bysio = anything2bytesio(src, sr=sr, volume=volume)
        return bysio.getvalue()


if __name__ == "__main__":
    print(__file__)
    inpath = r"../hello.wav"
    bys = anything2bytesio(inpath, sr=16000)
    print(bys)
    wav = anything2wav(bys, sr=16000)
    print(wav)
