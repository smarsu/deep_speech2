# Copyright 2019 smarsu. All Rights Reserved.

"""Interface used for train, test and value."""

import sys
import model
from datasets import datasets
from networks import deep_speech2, res_speech, shallow_speech


if __name__ == '__main__':
    if (len(sys.argv) <= 1):
        raise IOError('Usage: python {} train/value/test'.format(sys.argv[0]))

    parse = sys.argv[1]
    if parse == 'train': 
        root = '/share/datasets/data_aishell'
        aishell = datasets.Aishell(root)
        # net = deep_speech2.DeepSpeech2(201, 4231)
        # net = res_speech.ResSpeech()
        net = shallow_speech.ShallowSpeech()
        model = model.SpeechRecognitionModel(net, deep_speech2.ctc_loss, model_name='shallow_speech')
        model.train(aishell, epoch=1000000, batch_size=32, lr=0.1, params_path=None)
        # model.train(aishell, epoch=1000000, batch_size=32, lr=0.01, weight_decay=0., params_path='data/deep_speech2-0.01-4-2.2333707728450722')  # 18
        # model.train(aishell, epoch=1000000, batch_size=16, lr=0.001, weight_decay=0., params_path='data/deep_speech2-0.01-4-2.2333707728450722')  # 18
        # model.train(aishell, epoch=1000000, batch_size=32, lr=0.1 / 32, momentum=0., weight_decay=0., params_path=None)
    elif parse == 'value':
        root = '/share/datasets/data_aishell'
        aishell = datasets.Aishell(root)
        # net = deep_speech2.DeepSpeech2(201, 4231)
        net = shallow_speech.ShallowSpeech()
        model = model.SpeechRecognitionModel(net, deep_speech2.ctc_loss, device='cpu')
        model.value(aishell.train_datas(1, 'dev')[:100], params_path='data/shallow_speech-0.1-14-4.4856919112791225')
    elif parse == 'test':
        root = '/share/datasets/data_aishell'
        aishell = datasets.Aishell(root)
        net = deep_speech2.DeepSpeech2(201, 4231)
        # net = shallow_speech.ShallowSpeech()
        model = model.SpeechRecognitionModel(net, deep_speech2.ctc_loss)
        model.test(wav_path='test.wav', params_path='data/deep_speech2-0.01-4-2.2494239925925266', id2word=aishell.id2word)
