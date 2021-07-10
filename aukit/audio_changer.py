#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/12/22
"""
### audio_changer
变声器，变高低音，变语速，变萝莉音，回声。
基于librosa的变声。
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import numpy as np
import librosa

from .audio_io import _sr


def change_pitch(wav, sr=_sr, rate=0.):
    """
    调音高。
    :param rate:-20~20,float，0:原声
    :param wav:
    :param sr:
    :return:
    """
    return librosa.effects.pitch_shift(wav, sr=sr, n_steps=rate)


def change_speed(wav, sr=_sr, rate=0.):
    """
    调语速。
    :param rate:0~5,float，0:原声
    :param wav:
    :param sr:
    :return:
    """
    return librosa.effects.time_stretch(wav, rate)


def change_sample(wav, sr=_sr, rate=1):
    """
    调采样率，语速和音高同时改变。
    :param rate:0~5,float，1:原声
    :param wav:
    :param sr:
    :return:
    """
    return librosa.resample(wav, orig_sr=sr, target_sr=int(sr * rate))


def change_reback(wav, sr=_sr, rate=1):
    """
    回声。
    :param rate:1~10,int，1:原声
    :param wav:
    :param sr:
    :return:
    """
    frequencies, D = librosa.ifgram(wav, sr=sr)
    D = pool(D, size=(1, rate))
    D = repeat(D, rate)
    return librosa.istft(D)


def change_pitchspeed(wav, sr=_sr, rate=1):
    """
    音高和语速同时变化。
    :param rate:0~10,float，1:原声
    :param wav:
    :param sr:
    :return:
    """
    frequencies, D = librosa.ifgram(wav, sr=sr)
    n = int(D.shape[0] * rate)
    if n <= D.shape[0]:
        D = drop(D, D.shape[0] - n, mode="r")
    else:
        D = rewardshape(D, (n, D.shape[1]))
    return librosa.istft(D)


def change_attention(wav, sr=_sr, rate=0):
    """
    突出高音或低音段。
    :param rate:-100~100,int，0:原声
    :param wav:
    :param sr:
    :return:
    """
    frequencies, D = librosa.ifgram(wav, sr=sr)
    D = roll(D, rate)
    return librosa.istft(D)


def change_male(wav, sr=_sr, rate=0):
    """
    变男声。
    :param rate:0~1025,int，0,1,1025:原声
    :param wav:
    :param sr:
    :return:
    """
    frequencies, D = librosa.ifgram(wav, sr=sr)
    D = pool_step(D, rate)
    return librosa.istft(D)


def change_stretch(wav, sr=_sr, rate=1):
    """
    成倍拉伸延长。
    :param rate:1~10,int，1:原声
    :param wav:
    :param sr:
    :return:
    """
    frequencies, D = librosa.ifgram(wav, sr=sr)
    D = spread(D, rate)
    return librosa.istft(D)


def change_vague(wav, sr=_sr, rate=1):
    """
    模糊。
    :param rate:1~10,int，1:原声
    :param wav:
    :param sr:
    :return:
    """
    frequencies, D = librosa.ifgram(wav, sr=sr)
    D = pool(D, (1, rate))
    D = spread(D, (1, rate))
    return librosa.istft(D)


class CheckStep(object):
    def __init__(self, step):
        self.step = step
        self.index = 0

    def __call__(self, *args):
        self.index += 1
        return self.index % self.step != 0


def spread(D, size=(3, 3)):
    """传播，重复每个数据点。"""
    if isinstance(size, tuple):
        if size[0] > 1:
            D = np.repeat(D, size[0], axis=0)
        if size[1] > 1:
            D = np.repeat(D, size[1], axis=1)
        if size[0] * size[1] > 1:
            D = D / (size[0] * size[1])
    elif isinstance(size, int):
        D = np.repeat(D, size, axis=1)
    return D


def drop(D, n, mode="l"):
    """丢弃,mode:left,right,side,center"""
    if n == 0:
        return D
    if mode == "l":
        return D[n:]
    elif mode == "r":
        return D[:-n]
    elif mode == "s":
        return D[n // 2:-(n // 2)]
    elif mode == "c":
        if n < len(D):
            return np.vstack((D[:n // 2], D[-(n // 2):]))
        else:
            return ()
    else:
        raise AssertionError


def repeat(D, n, axis=0):
    """重复"""
    return np.repeat(D, n, axis=axis)


def roll(D, n):
    """循环移动"""
    return np.roll(D, n, axis=0)


def rewardshape(D, shape):
    """填充"""
    x = shape[0] - D.shape[0]
    y = shape[1] - D.shape[1]
    if x > 0:
        bottomlist = np.zeros([x, D.shape[1]])
        D = np.r_[D, bottomlist]
    if y > 0:
        rightlist = np.zeros([D.shape[0], y])
        D = np.c_[D, rightlist]
    return D


def pool_step(D, step):
    """步长池化"""
    _shape = D.shape
    if step < 2:
        return D
    cs = CheckStep(step)
    return rewardshape(np.array(list(filter(cs, D))), _shape)


def pool(D, size=(3, 3), shapeed=False):
    """池化"""
    _shape = D.shape
    if isinstance(size, tuple):
        if size[1] > 1:
            D = _pool(D, size[1])
        if size[0] > 1:
            D = _pool(D.T, size[0]).T
    elif isinstance(size, int):
        D = _pool(D.T, size).T
    if shapeed:
        D = rewardshape(D, _shape)
    return D


def _pool(D, poolsize):
    """池化方法"""
    x = D.shape[1] // poolsize
    restsize = D.shape[1] % poolsize
    if restsize > 0:
        x += 1
        rightlist = np.zeros([D.shape[0], poolsize - restsize])
        D = np.c_[D, rightlist]
    D = D.reshape((-1, poolsize))
    D = D.sum(axis=1).reshape(-1, x)
    return D


if __name__ == '__main__':
    from aukit.audio_player import play_audio

    path = r"E:/data/temp/01.wav"
    y, sr = librosa.load(path)

    # y = change_vague(3, y, sr)
    # y = change_pitch(-6, y, sr)
    # y = change_speed(0.5, y, sr)
    # y = change_sample(0.8, y, sr)
    # y = change_reback(3, y, sr)
    # y = change_pitchspeed(2, y, sr)
    # y = change_attention(50, y, sr)
    # y = change_male(5, y, sr)
    # y = change_vague(6, y, sr)

    """童声"""
    y = change_pitch(6, y, sr)
    y = change_male(20, y, sr)

    play_audio(y, sr=sr)
