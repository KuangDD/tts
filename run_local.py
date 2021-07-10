#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/12/1
"""
local
"""
from pathlib import Path
from functools import partial
from multiprocessing.pool import Pool
from matplotlib import pyplot as plt
# from tqdm import tqdm
import collections as clt
import os
import re
import json
import numpy as np
import shutil
import logging

logging.basicConfig(level=logging.INFO)


def run_spectrogram():
    from aukit import audio_spectrogram as asp
    from aukit import audio_griffinlim as agf
    from aukit import audio_io as aio
    from aukit.audio_player import play_audio
    inpath = r"hello.wav"
    wav, sr = aio.load_wav(inpath, with_sr=True)
    print(wav.shape, sr)
    # play_audio(wav, sr)

    lin_gf = agf.linear_spectrogram(wav)
    wav_gf = agf.inv_linear_spectrogram(lin_gf)
    # play_audio(wav_gf, sr)

    mel_sp = asp.mel_spectrogram(wav)
    mel_sp = asp.mel2linear_spectrogram(mel_sp)
    wav_sp = agf.inv_linear_spectrogram(mel_sp)
    # play_audio(wav_sp, sr)

    linear_gf = agf.linear_spectrogram(wav)
    mel_lin = agf.linear2mel_spectrogram(linear_gf)
    linear_mel = agf.mel2linear_spectrogram(mel_lin)
    wav_2 = agf.inv_linear_spectrogram(linear_mel)

    mel_sp = asp.mel_spectrogram(wav)
    mel_fea = asp.mel_spectrogram_feature(wav)

    # plt.figure()
    # plt.subplot("311")
    # plt.pcolor(linear)
    # plt.subplot("312")
    # plt.pcolor(linear2)
    # plt.subplot("313")
    # plt.pcolor(mel_fea)
    # plt.show()

    wav_ms = agf.inv_mel_spectrogram(mel_sp)
    wav_mf = agf.inv_mel_spectrogram(mel_fea)
    # play_audio(wav_ms, sr)
    # play_audio(wav_mf, sr)


def run_world():
    from aukit import audio_world as awd
    from aukit import audio_player as apr
    from aukit import audio_io as aio
    inpath = r"hello.wav"
    # sr, x = wavfile.read(inpath)
    x, sr = aio.load_wav(inpath, with_sr=True)
    f0, sp, ap = awd.world_spectrogram(x, sr)
    y = awd.inv_world_spectrogram(f0, sp, ap, sr)

    # apr.play_audio(x, sr)
    # apr.play_audio(y, sr)


def create_readme():
    from aukit import readme_docs
    docs = []
    with open("README.md", "wt", encoding="utf8") as fout:
        for doc in readme_docs:
            fout.write(doc)
            docs.append(doc)
    return "".join(docs)


def run_tuner():
    import aukit
    from aukit.audio_tuner import tune_speed, tune_pitch
    inpath = r"hello.wav"
    aukit.anything2bytes(inpath)
    aukit.anything2wav(inpath)
    aukit.anything2bytesio(inpath)
    bys = tune_speed(inpath, sr=16000, rate=0.5, out_type=None)
    print(bys)
    wav = tune_pitch(bys, sr=16000, rate=1, out_type=None)
    print(wav)
    # aukit.play_audio(wav)


def run_noise_remover():
    import aukit
    inpath = r"hello.wav"
    wav = aukit.load_wav(inpath)
    out = aukit.remove_noise(wav)
    # aukit.play_audio(out)


def run_player():
    import aukit
    inpath = Path(r"hello.wav")
    wav = aukit.load_wav(inpath, sr=16000)
    wav = aukit.change_voice(wav, mode="assign_pitch", alpha=200)
    # aukit.play_audio(wav, volume=0.5)


def run_aukit():
    import time
    t0 = time.time()
    from aukit.audio_io import __doc__ as io_doc
    from aukit.audio_editor import __doc__ as editor_doc
    from aukit.audio_tuner import __doc__ as tuner_doc
    from aukit.audio_player import __doc__ as player_doc
    from aukit.audio_noise_remover import __doc__ as noise_remover_doc
    from aukit.audio_normalizer import __doc__ as normalizer_doc
    from aukit.audio_spectrogram import __doc__ as spectrogram_doc
    from aukit.audio_griffinlim import __doc__ as griffinlim_doc
    from aukit.audio_changer import __doc__ as changer_doc
    from aukit.audio_cli import __doc__ as cli_doc
    from aukit.audio_world import __doc__ as world_doc
    t1 = time.time()
    print(t1 - t0)


def compare_hparams():
    from aukit.audio_griffinlim import default_hparams as gfhp
    from aukit.audio_spectrogram import default_hparams as sphp
    a = set(gfhp.items()) - set(sphp.items())
    b = set(sphp.items()) - set(gfhp.items())
    print(a)
    print(b)


def run_normalizer():
    import aukit
    from aukit.audio_player import play_sound
    from aukit import audio_normalizer as ano
    inpath = r"hello.wav"
    wav, sr = aukit.load_wav(inpath, with_sr=True)
    out = ano.remove_silence(wav)
    out = ano.tune_volume(wav, target_dBFS=-10)
    # play_sound(out, sr)


def run_editor():
    import aukit
    from aukit.audio_player import play_sound, play_audio
    from aukit import audio_editor as aed
    inpath = r"hello.wav"
    wav, sr = aukit.load_wav(inpath, with_sr=True)
    aud = aed.wav2audiosegment(wav, sr)
    out = aed.strip_audio(aud)
    wav = aed.audiosegment2wav(out)

    out = aed.remove_silence_wave(wav, sr=sr)
    out = aed.strip_silence_wave(out, sr=sr)

    print(len(wav), len(out))
    # play_audio(out, sr)


def convert_format(x):
    from aukit.audio_editor import convert_format_os

    return convert_format_os(**x)


def run_cli():
    from aukit.audio_cli import pool_jobs

    from pathlib import Path

    indir = Path(r"E:\lab\zhrtvc\data\samples\aishell")
    outdir = Path(r"E:\lab\zhrtvc\data\samples_wav\aishell")

    kw_lst = []
    for inpath in indir.glob("**/*.mp3"):
        parts = inpath.parent.relative_to(indir).parts
        name = "{}.{}".format(inpath.stem, 'wav')
        outpath = outdir.joinpath(*parts, name)
        outpath.parent.mkdir(exist_ok=True, parents=True)
        kw = dict(inpath=str(inpath), outpath=str(outpath), in_format=None, out_format='wav')
        kw_lst.append(kw)

    pool_jobs(func=convert_format, n_process=14, kwargs_list=kw_lst, tqdm_desc='convert_format')


if __name__ == "__main__":
    print(__file__)
    # run_spectrogram()
    # run_world()
    # create_readme()
    # run_tuner()
    # run_noise_remover()
    # run_player()
    # run_aukit()
    # compare_hparams()
    # run_normalizer()
    # run_editor()
    # run_cli()
