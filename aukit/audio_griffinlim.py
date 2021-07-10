#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/11/30
"""
### audio_griffinlim
griffinlim声码器，线性频谱转语音，梅尔频谱转语音，TensorFlow版本转语音，梅尔频谱和线性频谱相互转换。
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
from .audio_spectrogram import default_hparams
from .audio_io import Dict2Obj

try:
    import tensorflow as tf
    import lws
except ImportError as e:
    logger.info("ImportError: {}".format(e))

tmp = dict([('use_lws', False), ('frame_shift_ms', None), ('silence_threshold', 2), ('griffin_lim_iters', 30)])
default_hparams.update(tmp)


def hparams_debug_string(hparams=None):
    hparams = hparams or default_hparams
    values = hparams.values()
    hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
    return "Hyperparameters:\n" + "\n".join(hp)


def inv_linear_spectrogram(linear_spectrogram, hparams=None):
    """Converts linear spectrogram to waveform using librosa"""
    hparams = hparams or default_hparams
    if hparams.signal_normalization:
        D = _denormalize(linear_spectrogram, hparams)
    else:
        D = linear_spectrogram

    S = _db_to_amp(D + hparams.ref_level_db)  # Convert back to linear

    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)


def inv_mel_spectrogram(mel_spectrogram, hparams=None):
    """Converts mel spectrogram to waveform using librosa"""
    hparams = hparams or default_hparams
    if hparams.signal_normalization:
        D = _denormalize(mel_spectrogram, hparams)
    else:
        D = mel_spectrogram

    S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db), hparams)  # Convert back to linear

    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)


def inv_linear_spectrogram_tensorflow(linear_spectrogram, hparams=None):
    '''Builds computational graph to convert spectrogram to waveform using TensorFlow.
    Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
    inv_preemphasis on the output after running the graph.
    linear_spectrogram.shape[1] = n_fft
    '''
    hparams = hparams or default_hparams
    S = _db_to_amp_tensorflow(_denormalize_tensorflow(linear_spectrogram, hparams) + hparams.ref_level_db)
    return _griffin_lim_tensorflow(tf.pow(S, hparams.power), hparams)


def inv_linear_spectrogram_tf(linear_spectrogram, hparams=None):
    """
    返回wav语音信号。
    linear_spectrogram.shape[1] = num_freq = (n_fft / 2) + 1
    """
    hparams = hparams or default_hparams
    _shape = linear_spectrogram.shape
    tmp = np.concatenate(
        (linear_spectrogram, np.zeros((_shape[0], (hparams.n_fft // 2) + 1 - _shape[1]), dtype=np.float32)), axis=1)
    wav_tf = inv_linear_spectrogram_tensorflow(tmp, hparams)
    with tf.Session() as sess:
        return sess.run(wav_tf)


# 以下模块后续版本可能删除

def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    out = wav * 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, out.astype(np.int16))


def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


# From https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end


def get_hop_size(hparams=None):
    hparams = hparams or default_hparams
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    return hop_size


def linear_spectrogram(wav, hparams=None):
    hparams = hparams or default_hparams
    D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    S = _amp_to_db(np.abs(D), hparams) - hparams.ref_level_db

    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S


def mel_spectrogram(wav, hparams=None):
    hparams = hparams or default_hparams
    D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams), hparams) - hparams.ref_level_db

    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S


def mel_spectrogram_feature(wav, hparams=None):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    hparams = hparams or default_hparams
    frames = librosa.feature.melspectrogram(
        wav,
        hparams.sample_rate,
        n_fft=hparams.n_fft,
        hop_length=hparams.hop_size,
        n_mels=hparams.num_mels
    )
    return _amp_to_db(frames.astype(np.float32))


def linear2mel_spectrogram(linear_spectrogram, hparams=None):
    """Converts linear spectrogram to mel spectrogram"""
    hparams = hparams or default_hparams

    if hparams.signal_normalization:
        D = _denormalize(linear_spectrogram, hparams)
    else:
        D = linear_spectrogram

    D = _db_to_amp(D + hparams.ref_level_db)  # Convert back to linear

    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams), hparams) - hparams.ref_level_db

    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S


def mel2linear_spectrogram(mel_spectrogram, hparams=None):
    """Converts mel spectrogram to linear spectrogram"""
    hparams = hparams or default_hparams

    if hparams.signal_normalization:
        D = _denormalize(mel_spectrogram, hparams)
    else:
        D = mel_spectrogram

    D = _mel_to_linear(_db_to_amp(D - hparams.ref_level_db), hparams)  # Convert back to linear
    S = _amp_to_db(np.abs(D), hparams) - hparams.ref_level_db

    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S


def _lws_processor(hparams=None):
    hparams = hparams or default_hparams
    return lws.lws(hparams.n_fft, get_hop_size(hparams), fftsize=hparams.win_size, mode="speech")


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8, hparams=None):
    hparams = hparams or default_hparams
    window_length = int(hparams.sample_rate * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = _db_to_amp(threshold_db)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x:x + window_length]) < threshold:
            return x + hop_length
    return len(wav)


def _griffin_lim(S, hparams=None):
    """librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    """
    hparams = hparams or default_hparams
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)
    return y


def _griffin_lim_tensorflow(S, hparams=None):
    '''TensorFlow implementation of Griffin-Lim
    Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
    '''
    hparams = hparams or default_hparams
    with tf.variable_scope('griffinlim'):
        # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
        S = tf.expand_dims(S, 0)
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        y = _istft_tensorflow(S_complex, hparams)
        for i in range(hparams.griffin_lim_iters):
            est = _stft_tensorflow(y, hparams)
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            y = _istft_tensorflow(S_complex * angles, hparams)
        return tf.squeeze(y, 0)


def _stft(y, hparams=None):
    hparams = hparams or default_hparams
    if hparams.use_lws:
        return _lws_processor(hparams).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=hparams.win_size,
                            center=hparams.center)


def _stft_tensorflow(signals, hparams=None):
    hparams = hparams or default_hparams
    n_fft, hop_length, win_length = _stft_parameters(hparams)
    return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)


def _istft(y, hparams=None):
    hparams = hparams or default_hparams
    return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams.win_size, center=hparams.center)


def _istft_tensorflow(stfts, hparams=None):
    hparams = hparams or default_hparams
    n_fft, hop_length, win_length = _stft_parameters(hparams)
    return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


def _stft_parameters(hparams=None):
    hparams = hparams or default_hparams
    n_fft = hparams.n_fft  # (hparams.num_freq - 1) * 2
    hop_length = hparams.hop_size  # int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    win_length = hparams.win_size  # int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
    return n_fft, hop_length, win_length


##########################################################
# Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


##########################################################
# Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]


def _linear_to_mel(spectogram, hparams=None):
    hparams = hparams or default_hparams
    if hparams.mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    else:
        _mel_basis = hparams.mel_basis
    return np.dot(_mel_basis, spectogram)


def _mel_to_linear(mel_spectrogram, hparams=None):
    hparams = hparams or default_hparams
    if hparams.inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    else:
        _inv_mel_basis = hparams.inv_mel_basis
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _build_mel_basis(hparams=None):
    hparams = hparams or default_hparams
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels,
                               fmin=hparams.fmin, fmax=hparams.fmax)


def _amp_to_db(x, hparams=None):
    hparams = hparams or default_hparams
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def _normalize(S, hparams=None):
    hparams = hparams or default_hparams
    ma = hparams.max_abs_value
    mi = hparams.min_level_db
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return np.clip((2 * ma) * ((S - mi) / (-mi)) - ma, -ma, ma)
        else:
            return np.clip(ma * ((S - mi) / (-mi)), 0, ma)
    else:
        assert S.max() <= 0 and S.min() - mi >= 0
        if hparams.symmetric_mels:
            return (2 * ma) * ((S - mi) / (-mi)) - ma
        else:
            return ma * ((S - mi) / (-mi))


def _denormalize(D, hparams=None):
    hparams = hparams or default_hparams
    ma = hparams.max_abs_value
    mi = hparams.min_level_db
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return ((np.clip(D, -ma, ma) + ma) * -mi / (2 * ma)) + mi
        else:
            return (np.clip(D, 0, ma) * -mi / ma) + mi
    else:
        if hparams.symmetric_mels:
            return ((D + ma) * -mi / (2 * ma)) + mi
        else:
            return (D * -mi / ma) + mi


def _denormalize_tensorflow(S, hparams=None):
    hparams = hparams or default_hparams
    mi = hparams.min_level_db
    return (tf.clip_by_value(S, 0, 1) * -mi) + mi


if __name__ == "__main__":
    print(__file__)
