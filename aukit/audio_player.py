#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/11/23
"""
### audio_player
语音播放，传入文件名播放，播放wave数据，播放bytes数据。
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import sys
import time

from .audio_io import anything2bytesio, anything2wav
from .audio_io import _sr

try:
    import wave
    from pyaudio import PyAudio
    import sounddevice as sd
except ImportError as e:
    logger.info("ImportError: {}".format(e))


def play_audio(src=None, sr=_sr, volume=1.):
    chunk = 1024  # 2014kb
    bytesio = anything2bytesio(src, sr=sr, volume=volume)
    wf = wave.open(bytesio, "rb")
    p = PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True)
    t0 = time.time()
    while True:
        data = wf.readframes(chunk)
        if data == b"":
            break
        stream.write(data)
    stream.stop_stream()  # 停止数据流
    stream.close()
    p.terminate()  # 关闭 PyAudio
    t = time.time() - t0
    logger.info("play audio done, playing {:.2f} seconds.".format(t))


def play_sound(src, sr=_sr, **kwargs):
    data = anything2wav(src, sr=sr)
    t0 = time.time()
    sd.play(data, sr, **kwargs)
    sd.wait()
    t = time.time() - t0
    logger.info("play sound done, playing {:.2f} seconds.".format(t))


if __name__ == "__main__":
    print(__file__)
