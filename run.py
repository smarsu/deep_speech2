# Copyright 2019 smarsu. All Rights Reserved.

"""Interface used for train, test and value."""

import sys
import model
from datasets import datasets
from networks import deep_speech2, res_speech


if __name__ == '__main__':
    if (len(sys.argv) <= 1):
        raise IOError('Usage: python {} train/value/test'.format(sys.argv[0]))

    parse = sys.argv[1]
    if parse == 'train': 
        root = '/share/datasets/data_aishell'
        aishell = datasets.Aishell(root)
        # net = deep_speech2.DeepSpeech2(201, 4231)
        net = res_speech.ResSpeech()
        model = model.SpeechRecognitionModel(net, deep_speech2.ctc_loss, model_name='res_speech')
        # model.train(aishell, epoch=1000000, batch_size=32, lr=0.1)  # 18
        model.train(aishell, epoch=1000000, batch_size=32, lr=0.1, params_path=None)
    elif parse == 'value':
        root = '/share/datasets/data_aishell'
        aishell = datasets.Aishell(root)
        net = deep_speech2.DeepSpeech2(201, 4231)
        model = model.SpeechRecognitionModel(net, deep_speech2.ctc_loss)
        model.value(aishell.train_datas(1, 'dev'), params_path='data/deep_speech2-0.01-0-2.3161310894216753')
    elif parse == 'test':
        root = '/share/datasets/data_aishell'
        aishell = datasets.Aishell(root)
        net = deep_speech2.DeepSpeech2(201, 4231)
        model = model.SpeechRecognitionModel(net, deep_speech2.ctc_loss)
        model.test(wav_path='test.wav', params_path='data/deep_speech2-0.01-0-2.3161310894216753', id2word=aishell.id2word)
