# Copyright 2019 smarsu. All Rights Reserved.

"""Read and preprocess wave datas."""

import os.path as osp
import glog
import ctypes
import wave  # The wave package may cause memory leak
import numpy as np
import scipy.fftpack
import scipy.io.wavfile


def get_lib_path():
    rlpt = osp.realpath(__file__)
    rldir = osp.split(rlpt)[0]
    lbpt = osp.join(rldir, 'libwave.so')
    return lbpt


libwave = ctypes.cdll.LoadLibrary(get_lib_path())

ctypes.c_byte_p = ctypes.POINTER(ctypes.c_byte)
ctypes.c_short_p = ctypes.POINTER(ctypes.c_short)

class WaveSignal(object):
    def __init__(self, path):
        """
        Args:
            path: str.
        """
        self._nchannels = ctypes.c_int(0)
        self._sampwidth = ctypes.c_int(0)
        self._framerate = ctypes.c_int(0)
        self._nframes = ctypes.c_int(0)
        self._data = ctypes.c_void_p(0)

        pathInBytes = bytes(path, 'utf-8')

        libwave.ReadWave(ctypes.c_char_p(pathInBytes),  # you have to use bytes here.
                         ctypes.pointer(self._nchannels),
                         ctypes.pointer(self._sampwidth),
                         ctypes.pointer(self._framerate),
                         ctypes.pointer(self._nframes),
                         ctypes.pointer(self._data))

        # glog.info('{} info:'.format(path))
        # glog.info('\tnchannels: {}'.format(self.nchannels))
        # glog.info('\tsampwidth: {}'.format(self.sampwidth))
        # glog.info('\tframerate: {}'.format(self.framerate))
        # glog.info('\tnframes: {}'.format(self.nframes))
        # glog.info('\tdata: {}'.format(self._data))

    
    def __del__(self):
        # libwave.CptrFree(ctypes.cast(self._data, ctypes.c_void_p))
        pass  # We use buffer

    
    @property
    def nchannels(self):
        return self._nchannels.value


    @property
    def sampwidth(self):
        return self._sampwidth.value


    @property
    def framerate(self):
        return self._framerate.value


    @property
    def nframes(self):
        return self._nframes.value


    @property
    def data(self):
        if self.sampwidth == 8:
            self._data = ctypes.cast(self._data, ctypes.c_byte_p)
        elif self.sampwidth == 16:
            self._data = ctypes.cast(self._data, ctypes.c_short_p)
        else:
            raise ValueError('The supported sampwidth in wav is 8 or 16, '
                                'get {}'.format(sampwidth))
        return np.copy(np.ctypeslib.as_array(self._data, [self.nframes, self.nchannels]))


def open_wav(path):
    wave_signal = WaveSignal(path)
    return wave_signal.framerate, wave_signal.data


def read_wav_v2(wav_path):
    """Read wav datas.
    
    Reference:
        https://docs.python.org/3/library/wave.html

    Fix the dtype to int8 for normalize.

    Args:
        wav_path: str. The path of wave file.

    Returns:
        frames: ndarray, np.int8 or np.int16, shape [nchannels, nframes * sampwidth]. The 
            wav datas.
        framerate: int. The sampling frequency.
    """ 
    with wave.open(wav_path, mode='rb') as fb:
        nchannels, sampwidth, framerate, nframes, _, _ = fb.getparams()
        # glog.info('Read {} ... nchannels: {}, sampwidth: {}, framerate: {}, '
        #         'nframes: {}'.format(wav_path, nchannels, sampwidth, 
        #         framerate, nframes))
        wav = fb.readframes(nframes)

        frames = np.frombuffer(wav, dtype=np.int8)
        frames = frames.reshape(nframes * sampwidth, nchannels).T

        return frames, framerate


def read_wav(wav_path):
    """Read wav datas.
    
    Reference:
        https://docs.python.org/3/library/wave.html

    Args:
        wav_path: str. The path of wave file.

    Returns:
        frames: ndarray, np.int8 or np.int16, shape [nchannels, nframes]. The 
            wav datas.
        framerate: int. The sampling frequency.
    """
    fb = wave.open(wav_path, mode='rb')
    nchannels, sampwidth, framerate, nframes, _, _ = fb.getparams()
    glog.info('Read {} ... nchannels: {}, sampwidth: {}, framerate: {}, '
              'nframes: {}'.format(wav_path, nchannels, sampwidth, 
              framerate, nframes))
    wav = fb.readframes(nframes)
    fb.close()

    if sampwidth == 1:
        dtype = np.int8
    elif sampwidth == 2:
        dtype = np.int16
    else:
        raise ValueError('The supported sampwidth in wav is 1 or 2, '
                            'get {}'.format(sampwidth))
    frames = np.frombuffer(wav, dtype=dtype)
    frames = frames.reshape(nframes, nchannels).T

    # framerate, frames = scipy.io.wavfile.read(wav_path, False)
    # frames = frames.reshape(1, -1)
    # print(frames)

    # framerate, frames = open_wav(wav_path)
    # frames = frames.T
    # # print(frames)
    # # print()

    # assert np.sum(frames != frames) == 0

    # framerate = 16000
    # frames = np.ones(shape=[1, 16000 * 10])

    # frames = frames.reshape(nframes, nchannels).T
    # glog.info('Rate: {}, data shape: {}'.format(framerate, frames.shape))
    # print(frames)

    return frames, framerate


def get_frequency_feature_v2(wavsignal, framerate, feat_size=192):
    wavsignal = wavsignal[0]
    size = (len(wavsignal) + feat_size - 1) // feat_size * feat_size
    zeros = np.zeros(shape=[size,])
    zeros[:len(wavsignal)] = wavsignal
    return zeros.reshape(-1, feat_size)


def get_hamming_window(n, a):
    """Get the hamming window.
    
    Args:
        n: int
        a: int

    Returns:
        hamming_window: ndarray, np.float32, shape [n]
    """
    assert n > 1, 'In get hamming window, expect n > 1, get {}'.format(n)
    return (1 - a) - a * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def get_frequency_feature(wavsignal, framerate, time_window=25, time_stride=10):
    """Get the fft feature of wav signal.
    
    Reference:
        https://github.com/nl8590687/ASRT_SpeechRecognition/blob/master/general_function/file_wav.py#L148

    Args:
        wavsignal: ndarray, np.int8 or np.int16, shape [nchannels, nframes]. 
            The wav signal.
        framerate: int. The sampling frequency.
    
    Returns:
        windows: ndarray, np.float, shape [nwindows, feature_length]. The 
            fft feature of wavsignal.
    """
    if (framerate != 16000):
        raise ValueError('Currently only support wav audio files with '
                         'sampling rate of 160000Hz, but the audio is '
                         '{}Hz'.format(framerate))

    wavsignal = wavsignal[0]
    wav_length = len(wavsignal)

    time_window = 25  # ms
    time_stride = 10  # ms
    window_size = framerate // 1000 * time_window
    window_stride = framerate // 1000 * time_stride

    windows = []
    for i in range(window_size, wav_length + 1, window_stride):
        window = wavsignal[i-window_size:i]
        assert len(window) == window_size, 'the length of window not match.'

        window = window * get_hamming_window(window_size, 0.46)  # TODO(smarsu): calc once.
        window = scipy.fftpack.fft(window)
        window = window[:window_size//2+1]  # As the fft data is symmetrical, just choose half.

        window = np.abs(window) / wav_length  # TODO: Why divide wav_length
        window = np.log(window + 1)  # Will the hardware work well with the feature size (201)?

        windows.append(window)

    windows = np.stack(windows, 0)
    return windows


if __name__ == '__main__':
    while True:
        wavsignal, framerate = read_wav('/share/datasets/data_aishell/wav/train/S0002/BAC009S0002W0388.wav')
        # wavsignal, framerate = read_wav('test.wav')
        print('wavsignal:', wavsignal)
        print('framerate:', framerate)

        windows = get_frequency_feature(wavsignal, framerate)
        print('windows:', windows)
        print('windows shape:', windows.shape)
