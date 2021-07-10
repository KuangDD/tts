#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/12/15
"""
### audio_world
world声码器，提取语音的基频、频谱包络和非周期信号，频谱转为语音。调音高，调机器人音。
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import numpy as np
from .audio_io import _sr

try:
    import pyworld as pw
except ImportError as e:
    logger.info("ImportError: {}".format(e))


def world_spectrogram_default(wav, sr=_sr):
    """默认参数的world声码器语音转为特征频谱。"""
    # f0 : ndarray
    #     F0 contour. 基频等高线
    # sp : ndarray
    #     Spectral envelope. 频谱包络
    # ap : ndarray
    #     Aperiodicity. 非周期性
    f0, sp, ap = pw.wav2world(wav.astype(np.double), sr)  # use default options
    return f0, sp, ap


def inv_world_spectrogram_default(f0, sp, ap, sr=_sr):
    """默认参数的world声码器特征频谱转为语音。"""
    y = pw.synthesize(f0, sp, ap, sr)
    return y


def world_spectrogram(wav, sr=_sr, dim_num=32, **kwargs):
    """world声码器语音转为频谱。"""
    # 分布提取参数
    frame_period = kwargs.get("frame_period", pw.default_frame_period)
    f0_floor = kwargs.get("f0_floor", pw.default_f0_floor)
    f0_ceil = kwargs.get("f0_ceil", pw.default_f0_ceil)
    fft_size = kwargs.get("fft_size", pw.get_cheaptrick_fft_size(sr, f0_floor))
    ap_threshold = kwargs.get("ap_threshold", 0.85)
    f0_extractor = kwargs.get("f0_extractor", "dio")
    x = wav.astype(np.double)
    if f0_extractor == "dio":
        # 使用DIO算法计算音频的基频F0
        f0, t = pw.dio(x, sr, f0_floor=f0_floor, f0_ceil=f0_ceil)
    elif f0_extractor == "harvest":
        f0, t = pw.harvest(x, sr, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=frame_period)
    else:
        f0, t = f0_extractor(x, sr, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=frame_period)

    # 使用CheapTrick算法计算音频的频谱包络
    sp = pw.cheaptrick(x, f0, t, sr, f0_floor=f0_floor, fft_size=fft_size)
    # SP降维
    sp_enc = pw.code_spectral_envelope(sp, sr, number_of_dimensions=dim_num)

    # 计算aperiodic参数
    ap = pw.d4c(x, f0, t, sr, threshold=ap_threshold, fft_size=fft_size)
    # AP降维
    ap_enc = pw.code_aperiodicity(ap, sr)
    return f0, sp_enc, ap_enc


def inv_world_spectrogram(f0, sp, ap, sr=_sr, **kwargs):
    """world声码器频谱转为语音。"""
    frame_period = kwargs.get("frame_period", pw.default_frame_period)
    f0_floor = kwargs.get("f0_floor", pw.default_f0_floor)
    fft_size = kwargs.get("fft_size", pw.get_cheaptrick_fft_size(sr, f0_floor))
    sp_dec = pw.decode_spectral_envelope(sp, sr, fft_size=fft_size)
    ap_dec = pw.decode_aperiodicity(ap, sr, fft_size=fft_size)
    y = pw.synthesize(f0, sp_dec, ap_dec, sr, frame_period=frame_period)
    return y


def change_voice(wav, sr=_sr, mode="tune_pitch", alpha=1., fix=True):
    """
    变声。
    :param wav:
    :param sr:
    :param mode:
    :param alpha:
    :param fix:
    :return:
    """
    x = wav.astype(np.double)
    s = world_spectrogram_default(x, sr=sr)
    if mode == "tune_pitch":
        s = tune_pitch(*s, rate=alpha, fix=fix)
    elif mode == "tune_robot":
        s = tune_robot(*s, rate=alpha, fix=fix)
    elif mode == "assign_pitch":
        s = assign_pitch(*s, base=alpha, fix=fix)
    elif mode == "assign_robot":
        s = assign_robot(*s, base=alpha, fix=fix)
    else:
        logger.info("ModeError: mode={}".format(mode))
        raise TypeError
    y = inv_world_spectrogram_default(*s, sr=sr)
    return y


def tune_pitch(f0, sp, ap, rate=1., fix=False):
    """调音高"""
    f0_out = f0 * rate
    if fix:
        sp = fix_sp(sp, rate)
    return f0_out, sp, ap


def tune_robot(f0, sp, ap, rate=1., fix=False):
    """调机器人音"""
    tmp = f0[f0 > 0]
    if len(tmp) >= 1:
        m = np.percentile(tmp, 61.8)
    else:
        m = 1
    f0_out = np.ones_like(f0) * m * rate
    if fix:
        sp = fix_sp(sp, rate)
    return f0_out, sp, ap


def assign_pitch(f0, sp, ap, base=250, fix=False):
    """指定音高"""
    tmp = f0[f0 > 0]
    if len(tmp) >= 1:
        m = np.percentile(tmp, 61.8)
    else:
        m = 1
    rate = base / m
    f0_out = f0 * rate
    if fix:
        sp = fix_sp(sp, rate)
    return f0_out, sp, ap


def assign_robot(f0, sp, ap, base=250, fix=False):
    """指定音高的机器人音"""
    tmp = f0[f0 > 0]
    if len(tmp) >= 1:
        m = np.percentile(tmp, 61.8)
    else:
        m = 1
    rate = base / m
    f0_out = np.ones_like(f0) * m * rate
    if fix:
        sp = fix_sp(sp, rate)
    return f0_out, sp, ap


def fix_sp(sp, rate=1.):
    """修调频谱包络"""
    sp_dim = sp.shape[1]
    sp_out = np.zeros_like(sp)
    for f in range(sp_dim):
        f2 = min(sp_dim - 1, int(f / rate))
        sp_out[:, f] = sp[:, f2]
    return sp_out


if __name__ == "__main__":
    print(__file__)
