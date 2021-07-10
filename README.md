

![aukit](aukit.png "aukit")



## aukit

audio toolkit: 语音和频谱处理的工具箱。



### 安装



```

pip install -U aukit

```



- 注意

    * 可能需另外安装的依赖包：tensorflow, pyaudio, sounddevice。

    * tensorflow<=1.13.1

    * pyaudio暂不支持python37以上版本直接pip安装，需要下载whl文件安装，下载路径：https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

    * sounddevice依赖pyaudio。

    * aukit的默认音频采样率为16k。



### 版本

v1.4.5



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









### audio_changer

变声器，变高低音，变语速，变萝莉音，回声。

基于librosa的变声。



### audio_editor

语音编辑，切分音频，去除语音中的较长静音，去除语音首尾静音，设置采样率，设置通道数。

音频格式相互转换，例如wav格式转为mp3格式。

切分音频，去除静音，去除首尾静音输入输出都支持wav格式。

语音编辑功能基于pydub的方法，增加了数据格式支持。



### audio_griffinlim

griffinlim声码器，线性频谱转语音，梅尔频谱转语音，TensorFlow版本转语音，梅尔频谱和线性频谱相互转换。



### audio_io

语音IO，语音保存、读取，支持wav和mp3格式，语音形式转换（np.array,bytes,io.BytesIO），支持【.】操作符的字典。



### audio_noise_remover

语音降噪，降低环境噪声。



### audio_normalizer

语音正则化，去除音量低的音频段（去除静音），调节音量。

语音正则化方法基于VAD的方法。



### audio_player

语音播放，传入文件名播放，播放wave数据，播放bytes数据。



### audio_spectrogram

语音频谱，语音转线性频谱，语音转梅尔频谱。



### audio_tuner

语音调整，调整语速，调整音高。



### audio_world

world声码器，提取语音的基频、频谱包络和非周期信号，频谱转为语音。调音高，调机器人音。



### 历史版本

#### v1.4.5

- logger改为__file__。



#### v1.4.4

- Dict2Obj方法增加parse方法。



#### v1.4.3

- 修正Dict2Obj的get方法达不到预期的bug。

- 修正world声码器变声静音报错的bug。



#### v1.4.1

- 修正安装依赖报错的bugs。

- set系列改为convert系列，如set_sample_rate改为convert_sample_rate。

- Dict2Obj用dotmap模块的DotMap代替。



#### v1.4.0

- 增加音频格式转换方法。

- 命令行批量处理语音，支持音频播放、语音降噪、音频格式转换。

- 增加命令行工具：auplay, aunoise, auformat。

- 调整日志输出方法。



#### v1.3.12

- 减少不必要的依赖，最低依赖只保留pydub,  scipy, numpy, librosa。

