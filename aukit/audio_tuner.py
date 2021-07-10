#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: KDD
# @time: 2018-11-10
"""
### audio_tuner
语音调整，调整语速，调整音高。
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

from pydub import AudioSegment
from scipy.io import wavfile
import numpy as np
import io
from .audio_io import anything2bytesio, anything2wav
from .audio_io import _sr


def tune_speed(src=None, sr=_sr, rate=1., out_type=np.ndarray):
    """
    变语速
    rate = win / (bar - cro)
    :param src:
    :param rate:
    :return:
    """
    song = AudioSegment.from_wav(anything2bytesio(src, sr=sr))
    n_song = len(song)
    win = 50
    bar = 100
    cro = int(bar - win / rate)

    segs = []
    for i in range(0, n_song - bar, win):
        segs.append(song[i: i + bar])

    out_song = segs[0]
    for seg in segs[1:]:
        out_song = out_song.append(seg, cro)

    io_out = io.BytesIO()
    out_song.export(io_out, format="wav")

    if out_type is np.ndarray:
        return anything2wav(io_out.getvalue(), sr=sr)
    else:
        return anything2bytesio(io_out.getvalue(), sr=sr)


def tune_pitch(src=None, sr=_sr, rate=1., out_type=np.ndarray):
    """
    变音调
    :param io_in:
    :param rate:
    :return:
    """
    frate, wavdata = wavfile.read(anything2bytesio(src, sr=sr))

    cho_ids = [int(w) for w in np.arange(0, len(wavdata), rate)]
    out_wavdata = wavdata[cho_ids]

    io_out = io.BytesIO()
    wavfile.write(io_out, frate, out_wavdata)
    io_out = tune_speed(io_out, rate=1 / rate, out_type=io.BytesIO)
    if out_type is np.ndarray:
        return anything2wav(io_out.getvalue(), sr=sr)
    else:
        return anything2bytesio(io_out.getvalue(), sr=sr)


if __name__ == "__main__":
    print(__file__)
