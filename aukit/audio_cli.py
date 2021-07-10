#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/1/5
"""
### audio_cli
命令行，播放音频，去除背景噪声，音频格式转换。
支持递归处理文件夹内的全部音频。

#### 命令行

##### **说明**

- 用位置参数来控制。
- 名称说明
    * inpath：输入音频路径或目录。
    * outpath：输出音频路径或目录，如果为目录，则输出的子目录按照inpath的子目录格式输出。
    * sr：音频采样率，默认16000或自动识别采样率。
    * in_format：输入音频格式，主要用以限制为指定后缀名的文件，如果不设置，则处理目录的全部文件。
    * out_format：输出音频格式，主要用以音频格式转换，设置输出音频的后缀名。
- 中括号【[]】里面的是可选参数。

#### **工具**
- auplay: 播放音频

```
auplay inpath [sr] [in_format]
```

- aunoise: 语音降噪

```
aunoise inpath outpath [in_format]
```


- auformat: 音频格式转换

```
auformat inpath outpath out_format [in_format]
```



"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import sys
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
import traceback
import multiprocessing as mp

from .audio_io import _sr
from .audio_player import play_audio
from .audio_noise_remover import remove_noise_os
from .audio_editor import convert_format_os

_n_process = max(1, mp.cpu_count() - 2)


def play_audio_one(inpath, sr):
    try:
        play_audio(inpath, sr=sr)
    except Exception as e:
        logger.info("PlayAudioFailed: {}".format(inpath))
        traceback.print_exc()


def play_audio_cli():
    fpath = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        sr = int(sys.argv[2])
    else:
        sr = _sr
    if len(sys.argv) >= 4:
        in_format = sys.argv[3]
    else:
        in_format = None

    if fpath.is_file():
        play_audio_one(fpath, sr=sr)
    elif fpath.is_dir():
        tmp = "**/*" if in_format is None else "**/*.{}".format(in_format)
        for inpath in sorted(fpath.glob(tmp)):
            if not inpath.is_file():
                continue
            try:
                play_audio_one(inpath, sr=sr)
            except Exception as e:
                logger.info("PlayAudioFailed: {}".format(inpath))
                traceback.print_exc()
    else:
        assert fpath.exists()


def remove_noise_one(x):
    outpath = Path(x["outpath"])
    outpath.parent.mkdir(exist_ok=True, parents=True)
    try:
        out = remove_noise_os(**x)
    except Exception as e:
        inpath = str(x["inpath"])
        logger.info("RemoveNoiseFailed: {}".format(inpath))
        traceback.print_exc()
        out = None
    return out


def remove_noise_cli():
    inpath = Path(sys.argv[1])
    outpath = Path(sys.argv[2])
    if len(sys.argv) >= 4:
        in_format = sys.argv[3]
    else:
        in_format = None

    if inpath.is_file():
        remove_noise_one(dict(inpath=inpath, outpath=outpath))
    elif inpath.is_dir():
        indir = inpath
        outdir = outpath
        kw_lst = []
        tmp = "**/*" if in_format is None else "**/*.{}".format(in_format)
        for inpath in indir.glob(tmp):
            if not inpath.is_file():
                continue
            parts = inpath.relative_to(indir).parts
            outpath = outdir.joinpath(*parts)
            kw = dict(inpath=str(inpath), outpath=str(outpath))
            kw_lst.append(kw)

        pool_jobs(func=remove_noise_one, n_process=_n_process, kwargs_list=kw_lst, tqdm_desc='remove_noise')
    else:
        assert inpath.exists()


def convert_format_one(x):
    outpath = Path(x["outpath"])
    outpath.parent.mkdir(exist_ok=True, parents=True)
    try:
        out = convert_format_os(**x)
    except Exception as e:
        inpath = str(x["inpath"])
        logger.info("ConvertFormatFailed: {}".format(inpath))
        traceback.print_exc()
        out = None
    return out


def convert_format_cli():
    inpath = Path(sys.argv[1])
    outpath = Path(sys.argv[2])
    out_format = sys.argv[3]
    if len(sys.argv) >= 5:
        in_format = sys.argv[4]
    else:
        in_format = None

    if inpath.is_file():
        convert_format_one(dict(inpath=inpath, outpath=outpath, in_format=in_format, out_format=out_format))
    elif inpath.is_dir():
        indir = inpath
        outdir = outpath
        kw_lst = []
        tmp = "**/*" if in_format is None else "**/*.{}".format(in_format)
        for inpath in indir.glob(tmp):
            if not inpath.is_file():
                continue
            parts = inpath.parent.relative_to(indir).parts
            name = "{}.{}".format(inpath.stem, out_format)
            outpath = outdir.joinpath(*parts, name)
            kw = dict(inpath=str(inpath), outpath=str(outpath), in_format=in_format, out_format=out_format)
            kw_lst.append(kw)

        pool_jobs(func=convert_format_one, n_process=_n_process, kwargs_list=kw_lst, tqdm_desc='convert_format')
    else:
        assert inpath.exists()


def pool_jobs(func, n_process=_n_process, kwargs_list=(), post_func=None, tqdm_desc="job"):
    """
    多进程执行任务。
    :param func: 第一个参数为变化参数，如果多个参数组合变化，请用：【def _func(x): return func(**x)】的方式处理函数。
    :param n_process:
    :param kwargs_list:
    :param post_func:
    :param tqdm_desc:
    :return:
    """
    if post_func is None:
        post_func = lambda x: x
    _tqdm = lambda x: tqdm(x, desc=tqdm_desc, total=len(kwargs_list), ncols=100, mininterval=1)
    if n_process == 0 or n_process == 1:
        for kw in _tqdm(kwargs_list):
            out = func(kw)
            post_func(out)
    else:
        partial_func = partial(func)
        job = Pool(n_process).imap(partial_func, kwargs_list)
        for out in list(_tqdm(job)):
            post_func(out)


if __name__ == "__main__":
    print(__file__)
