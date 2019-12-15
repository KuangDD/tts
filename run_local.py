#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/12/1
"""
"""
from pathlib import Path
from functools import partial
from multiprocessing.pool import Pool
from matplotlib import pyplot as plt
from tqdm import tqdm
import collections as clt
import os
import re
import json
import numpy as np
import shutil


def run_spectrogram():
    from aukit import audio_spectrogram as asp
    from aukit import audio_griffinlim as agf
    from aukit import audio_io as aio
    from aukit.audio_player import play_audio
    inpath = r"E:/data/temp/01.wav"
    wav, sr = aio.load_wav(inpath, with_sr=True)
    print(wav.shape, sr)
    mel_gf = agf.mel_spectrogram(wav)
    mel_sp = asp.mel_spectrogram(wav)
    mel_fea = asp.mel_spectrogram_feature(wav)

    plt.figure()
    plt.subplot("311")
    plt.pcolor(mel_gf)
    plt.subplot("312")
    plt.pcolor(mel_sp)
    plt.subplot("313")
    plt.pcolor(mel_fea)
    plt.show()

    wav_mg = agf.inv_mel_spectrogram(mel_gf)
    wav_ms = agf.inv_mel_spectrogram(mel_sp)
    wav_mf = agf.inv_mel_spectrogram(mel_fea)
    play_audio(wav_mg, sr)


def run_world():
    from scipy.io import wavfile
    from aukit import audio_world as awd
    from aukit import audio_player as apr
    from aukit import audio_io as aio
    inpath = r"E:/data/temp/01.wav"
    # sr, x = wavfile.read(inpath)
    x, sr = aio.load_wav(inpath, with_sr=True)
    f0, sp, ap = awd.world_spectrogram(x, sr)
    y = awd.inv_world_spectrogram(f0, sp, ap, sr)
    apr.play_audio(x, sr)
    apr.play_audio(y, sr)


if __name__ == "__main__":
    print(__file__)
    # run_spectrogram()
    run_world()