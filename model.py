# Copyright 2019 smarsu. All Rights Reserved.

"""Main for train, test and value."""

import time
import os.path as osp
import glog
from tqdm import tqdm
import torch
import numpy as np
from process import process_voice, postprocess_ctc
# from datasets import datasets
from networks import deep_speech2
from metrics import character_error_rate


class SpeechRecognitionModel(object):
    """The base class of speech recognition model."""
    def __init__(self, 
                 model, 
                 criterion,
                 model_name='deep_speech2',
                 device='gpu'):
        if device == 'gpu':
            self.model = model.cuda()
        elif device == 'cpu':
            self.model = model
        # self.criterion = criterion
        self._model_name = model_name
        pass


    def _build_model(self):
        """If the model if pytorch model, we do not need this."""
        pass


    # def _preprocess(self, wav_paths):
    #     return np.ones([len(wav_paths), 1, 1000, 201], dtype=np.float32), [30] * len(wav_paths)


    def _preprocess_v2(self, wav_paths):
        """For res_speech."""
        minibatch_windows = []
        max_window_size = 0
        window_sizes = []
        for wav_path in wav_paths:
            wavsignal, framerate = process_voice.read_wav_v2(wav_path)
            wavsignal = process_voice.norm(wavsignal)  # norm should before get_frequency_feature_v2
            windows = process_voice.get_frequency_feature_v2(wavsignal, framerate)
            max_window_size = max(max_window_size, windows.shape[0])
            minibatch_windows.append(windows)
            # window_sizes.append(deep_speech2.calc_downsampled_t_length(windows.shape[0]))
            # Warning: The right the window_sizes, the quick the loss get down.
            window_sizes.append(self.model.calc_t_length(windows.shape[0]))  # t // 32

        minibatch_input = np.zeros(shape=[len(wav_paths), 
                                          max_window_size, 
                                          minibatch_windows[0].shape[1]], dtype=np.float32)  # for simple, use 0 for pad.
        for i in range(len(wav_paths)):  # TODO: pad both side.
            minibatch_input[i:i+1, :len(minibatch_windows[i])] = minibatch_windows[i]
        
        return minibatch_input[:, np.newaxis, ...].transpose(0, 3, 2, 1), window_sizes


    def _preprocess(self, wav_paths):
        """
        Args:
            wav_path: list of str.

        Returns:
            minibatch_input: ndarray, shape [N, 1, T, C].
        """
        minibatch_windows = []
        max_window_size = 0
        window_sizes = []
        for wav_path in wav_paths:
            wavsignal, framerate = process_voice.read_wav(wav_path)
            windows = process_voice.get_frequency_feature(wavsignal, framerate, time_window=52, time_stride=52)
            # windows = process_voice.norm(windows)
            max_window_size = max(max_window_size, windows.shape[0])
            minibatch_windows.append(windows)
            window_sizes.append(self.model.calc_t_length(windows.shape[0]))

        minibatch_input = np.zeros(shape=[len(wav_paths), 
                                          max_window_size, 
                                          minibatch_windows[0].shape[1]], dtype=np.float32)  # for simple, use 0 for pad.
        for i in range(len(wav_paths)):  # TODO: pad both side.
            minibatch_input[i:i+1, :len(minibatch_windows[i])] = minibatch_windows[i]
        
        return minibatch_input[:, np.newaxis, ...].transpose(0, 3, 2, 1), window_sizes


    def _postprocess_per_batch(self, predict):
        """
        Args:
            predict: ndarray, shape [N, T, C]. The output of self.model. Make sure the dim of batch is 1.
        """
        assert predict.shape[0] == 1, 'The dim of batch in predict should be ' \
                                      '1, not {}'.format(predict.shape[0])

        return postprocess_ctc.ctc_greedy_decoder(predict, [predict.shape[1]])


    def _postprocess(self, predict):
        """
        Args:
            predict: The output of self.model.

        Returns:
        
        """
        sequences = []
        for i in range(len(predict)):
            sequence = self._postprocess_per_batch(predict[i:i+1])
            sequences.extend(sequence)
        return sequences


    def _pad_label(self, labels):
        """
        Args:
            labels: list of list. The inner list with difference length.

        Returns:

        """
        lengths = [len(label) for label in labels]
        max_length = max(lengths)
        labels = [label + [0] * (max_length - len(label)) for label in labels]
        return labels, lengths

    
    def train(self, dataset, epoch, batch_size, lr=0.1, momentum=0.9, weight_decay=1e-6, params_path=None):
        """
        Args:
            dataser: The object of Dataset. We use the train_datas function 
                to get batch datas.
            batch_size: int, >0
            lr: float
            weight_decay: float
        """
        self.model = self.model.train()
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(self.model.parameters())
        ctc_loss = torch.nn.CTCLoss(reduction='mean')

        if params_path:
            self.model.load_state_dict(torch.load(params_path))
            glog.info('Load params ... {}'.format(params_path))

        for step in range(epoch):
            running_loss = 0.
            run_cnt = 0
            pbar = dataset.train_datas(batch_size, 'train', limited_data_size=None)
            # pbar = tqdm(dataset.train_datas(batch_size, 'train', limited_data_size=None))
            for idx, data_tuple in enumerate(pbar):
                t1 = time.time()
                assert data_tuple.shape == (batch_size, 2)
                data = data_tuple[:, 0]
                data, window_sizes = self._preprocess(data)
                label = data_tuple[:, 1]
                input_lengths = torch.from_numpy(np.array(window_sizes, dtype=np.int32))
                # print(window_sizes)

                label, lengths = self._pad_label(label)
                target_lengths = torch.from_numpy(np.array(lengths, dtype=np.int32))

                # print(lengths)

                # if max(input_lengths) > 40:
                #     glog.info('continue {} > 40'.format(input_lengths))
                #     continue

                if (np.sum(np.array(input_lengths) < np.array(target_lengths)) != 0):
                    glog.info('continue {}/{}, processing ... {}/{}'.format(
                        input_lengths, 
                        target_lengths, 
                        idx, 
                        len(pbar)))
                    continue

                optimizer.zero_grad()

                t2 = time.time()

                print(data.shape)

                input = torch.from_numpy(data).cuda()
                
                predict = self.model(input)
                predict = predict.permute(1, 0, 2)
                predict = predict.log_softmax(2)
                loss = ctc_loss(predict, 
                                torch.from_numpy(np.array(label)).cuda(), 
                                input_lengths.cuda(), 
                                target_lengths.cuda())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                run_cnt += 1

                t3 = time.time()

                # print(predict.cpu().detach().numpy())
                # print(input_lengths.cpu().numpy())
                # print(target_lengths.cpu().numpy())
                glog.info('loss: {}, processing ... {}/{}, tpre: {}, tmodel: {}'.format(
                    running_loss / run_cnt, idx, len(pbar), t2 - t1, t3 - t2))
                # print()

            torch.save(self.model.state_dict(), 
                        'data/' + '-'.join([self._model_name, 
                                            str(lr), 
                                            str(step), 
                                            str(running_loss / (idx + 1))]))

            # self.value(dataset.train_datas(batch_size, 'dev', limited_data_size=1000), params_path=None)


    def value(self, data_tuples, params_path):
        """Value the performance on dev datasets.
        
        Args:
            data_tuples: list of tuple, shape [N, b, 2], [(data, label), ...].
        """
        self.model = self.model.eval().cpu()
        if params_path:
            self.model.load_state_dict(torch.load(params_path))

        preds = []
        labels = []
        for data_tuple in data_tuples:
            data = data_tuple[:, 0]
            label = data_tuple[:, 1]
            data, _ = self._preprocess(data)
            # data = data / 127
            print(data.shape)
            predict = self.model(torch.from_numpy(data)).cpu().detach().numpy()
            sequence = self._postprocess(predict)
            preds.extend(sequence)
            labels.extend(label)
        
        print(preds)
        cer = character_error_rate.cer(preds, labels)
        glog.info('cer in dev set: {}'.format(cer))


    def test(self, wav_path, params_path, id2word):
        """Test with a wave file.
        
        Args:
            wav_path: str.
            params_path: str.
        """
        self.model = self.model.eval()
        self.model.load_state_dict(torch.load(params_path))

        data, _ = self._preprocess([wav_path])
        predict = self.model(torch.from_numpy(data).cuda()).cpu().detach().numpy()
        sequence = self._postprocess(predict)[0]

        words = ''.join([id2word[id] for id in sequence])
        glog.info('pred sequence: {}'.format(words))
