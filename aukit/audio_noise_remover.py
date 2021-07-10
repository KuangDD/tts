#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/11/23
"""
### audio_noise_remover
语音降噪，降低环境噪声。
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import numpy as np
import ctypes as ct
from .audio_io import load_wav, save_wav
from .audio_io import _sr
import traceback


def remove_noise(wav: np.array, sr=_sr, **kwargs):
    """
    谱减法去除背景噪声。
    :param wav: 语音信号
    :param sr: 采样率
    :param kwargs:
    :return: np.ndarray
    """
    x = wav
    noise_span = kwargs.get("noise_span", (0, 100))
    noise_wav = kwargs.get("noise_wav", None)
    threshold = kwargs.get("threshold", 3)
    beta = kwargs.get("beta", 0.002)
    # 计算参数
    unit_ = 20  # 每帧时长，单位ms
    len_ = unit_ * sr // 1000  # 样本中帧的大小
    PERC = 50  # 窗口重叠占帧的百分比
    len1 = len_ * PERC // 100  # 重叠窗口
    len2 = len_ - len1  # 非重叠窗口
    # 设置默认参数
    Thres = threshold
    Expnt = 2.0
    beta = beta
    G = 0.9
    # 初始化汉明窗
    win = np.hamming(len_)
    # normalization gain for overlap+add with 50% overlap
    winGain = len2 / sum(win)

    # Noise magnitude calculations - assuming that the first 5 frames is noise/silence
    nFFT = 2 * 2 ** (nextpow2(len_))
    noise_mean = np.zeros(nFFT)
    if noise_wav is None:
        sidx = noise_span[0] // unit_
        eidx = noise_span[1] // unit_
        for k in range(sidx, eidx):
            noise_mean = noise_mean + abs(np.fft.fft(win * x[k * len_:(k + 1) * len_], nFFT))
        noise_mu = noise_mean / (eidx - sidx)
    else:
        if "noise_span" in kwargs:
            sidx = noise_span[0] // unit_
            eidx = noise_span[1] // unit_
        else:
            sidx = 0
            eidx = len(noise_wav) // unit_
        for k in range(sidx, eidx):
            noise_mean = noise_mean + abs(np.fft.fft(win * x[k * len_:(k + 1) * len_], nFFT))
        noise_mu = noise_mean / (eidx - sidx)
    # --- allocate memory and initialize various variables
    k = 1
    img = 1j
    x_old = np.zeros(len1)
    Nframes = len(x) // len2 - 1
    xfinal = np.zeros(Nframes * len2)

    # =========================    Start Processing   ===============================
    for n in range(0, Nframes):
        # Windowing
        insign = win * x[k - 1:k + len_ - 1]
        # compute fourier transform of a frame
        spec = np.fft.fft(insign, nFFT)
        # compute the magnitude
        sig = abs(spec)

        # save the noisy phase information
        theta = np.angle(spec)
        SNRseg = 10 * np.log10(np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)

        if Expnt == 1:  # 幅度谱
            alpha = berouti1(SNRseg)
        else:  # 功率谱
            alpha = berouti(SNRseg)
        #############
        sub_speech = sig ** Expnt - alpha * noise_mu ** Expnt
        # 当纯净信号小于噪声信号的功率时
        diffw = sub_speech - beta * noise_mu ** Expnt

        # beta negative components

        z = find_index(diffw)
        if len(z) > 0:
            # 用估计出来的噪声信号表示下限值
            sub_speech[z] = beta * noise_mu[z] ** Expnt
            # --- implement a simple VAD detector --------------
        if SNRseg < Thres:  # Update noise spectrum
            noise_temp = G * noise_mu ** Expnt + (1 - G) * sig ** Expnt  # 平滑处理噪声功率谱
            noise_mu = noise_temp ** (1 / Expnt)  # 新的噪声幅度谱
        # flipud函数实现矩阵的上下翻转，是以矩阵的“水平中线”为对称轴
        # 交换上下对称元素
        sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])
        x_phase = (sub_speech ** (1 / Expnt)) * (
                np.array([np.cos(x) for x in theta]) + img * (np.array([np.sin(x) for x in theta])))
        # take the IFFT

        xi = np.fft.ifft(x_phase).real
        # --- Overlap and add ---------------
        xfinal[k - 1:k + len2 - 1] = x_old + xi[0:len1]
        x_old = xi[0 + len1:len_]
        k = k + len2
    return winGain * xfinal


def remove_noise_os(inpath, outpath, **kwargs):
    try:
        wav, sr = load_wav(inpath, with_sr=True)
        out = remove_noise(wav, sr, **kwargs)
        save_wav(out, outpath, sr)
    except Exception as e:
        logger.info('Error path: {}'.format(inpath))
        logger.info('Error info: {}'.format(e))
        traceback.print_exc()


class FloatBits(ct.Structure):
    _fields_ = [
        ('M', ct.c_uint, 23),
        ('E', ct.c_uint, 8),
        ('S', ct.c_uint, 1)
    ]


class Float(ct.Union):
    _anonymous_ = ('bits',)
    _fields_ = [
        ('value', ct.c_float),
        ('bits', FloatBits)
    ]


def nextpow2(x):
    if x < 0:
        x = -x
    if x == 0:
        return 0
    d = Float()
    d.value = x
    if d.M == 0:
        return d.E - 127
    return d.E - 127 + 1


def berouti(SNR):
    if -5.0 <= SNR <= 20.0:
        a = 4 - SNR * 3 / 20
    elif SNR < -5.0:
        a = 5
    else:
        a = 1
    return a


def berouti1(SNR):
    if -5.0 <= SNR <= 20.0:
        a = 3 - SNR * 2 / 20
    elif SNR < -5.0:
        a = 4
    else:
        a = 1
    return a


def find_index(x_list):
    index_list = []
    for i in range(len(x_list)):
        if x_list[i] < 0:
            index_list.append(i)
    return index_list


if __name__ == "__main__":
    print(__file__)
