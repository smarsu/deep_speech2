# Copyright 2019 smarsu. All Rights Reserved.

"""Load speech recognition datasets."""

import os
import os.path as osp
import glog
import numpy as np

os.environ['PYTHONIOENCODING'] = 'utf-8'


def save_dict(dictionary, path):
    """Save the dictionary to disk as the data may be disorder in the python 
        dict.
    
    Args:
        dictionary: list.
    """
    with open(path, 'w', encoding='utf8') as fb:
        for word in dictionary:
            fb.write(word + '\n')


def load_dict(path):
    """Load the dictionary from disk.
    
    Args:
        path: str.

    Returns:
        id2word: dict
        word2id: dict
    """
    with open(path, 'r', encoding='utf8') as fb:
        lines = fb.readlines()
        words = [line.strip() for line in lines]

    id2word = {idx:word for idx, word in enumerate(words)}
    word2id = {word:idx for idx, word in enumerate(words)}
    return id2word, word2id


def build_dict(sentences, saved_path=None):
    """Build the dictionary based on the train sentences.
    
    Args:
        sentences: list of str.
        saved_path: str or None. If not None, save the dictionary to disk

    Returns:
        id2word: dict
        word2id: dict
    """
    dictionary = {}
    for sentence in sentences:
        for word in sentence:
            dictionary[word] = dictionary.get(word, 0) + 1
    dictionary = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    dictionary = [('<BLACK>', 0)] + dictionary  # black for ctx loss
    if saved_path is not None:
        save_dict([word for word, _ in dictionary], saved_path)

    id2word = {idx:word for idx, (word, _) in enumerate(dictionary)}
    word2id = {word:idx for idx, (word, _) in enumerate(dictionary)}
    return id2word, word2id


class Dataset(object):
    """Base dataset class."""
    def __init__(self):
        pass


    def train_datas(self, batch_size=1, mode='train', limited_data_size=None):
        """Get the train datas, include wave paths and labels.

        The train datas will be shuffled. (The paper suggest sort by label 
        length at the first epoch)

        And the train_datas will be reshaped to match the batchsize
        
        Args:
            batch_size: int. 

        Returns:
            train_datas: ndarray, shape [n, batch_size, 2]
        """
        # list of tuple, [(wave_path, sentences), ...]
        if mode == 'train':
            datas = self._train_tuples
        elif mode == 'dev':
            datas = self._dev_tuples
        elif mode == 'test':
            datas = self._test_tuples

        if limited_data_size:
            datas = datas[:limited_data_size]

        datas_length = len(datas)
        shuffled_ids = np.arange(datas_length)
        np.random.shuffle(shuffled_ids)

        shuffled_ids = shuffled_ids[:datas_length//batch_size*batch_size]
        shuffled_ids = shuffled_ids.reshape(-1, batch_size)

        datas = np.array(datas)
        datas = datas[shuffled_ids]

        return datas


class Aishell(Dataset):
    """aishell speech recoginition dataset."""
    def __init__(self, root):
        """Load aishell dataset.

        The tree of root as the follow:
        .
        |-- transcript
        `-- wav

        Args:
            root: str.
        """
        self._label_path = self._get_label_path(root)
        self._name2words = self._parse_label_path(self._label_path)

        self._train_paths = self._get_wav_paths(root, 'train')
        self._dev_paths = self._get_wav_paths(root, 'dev')
        self._test_paths = self._get_wav_paths(root, 'test')

        self._train_tuples = self._get_data_tuple(self._train_paths, self._name2words, 'train')
        self._dev_tuples = self._get_data_tuple(self._dev_paths, self._name2words, 'dev')
        self._test_tuples = self._get_data_tuple(self._test_paths, self._name2words, 'test')

        saved_path_dictionary = osp.join(root, 'dictionary.txt')
        if osp.exists(saved_path_dictionary):
            self.id2word, self.word2id = load_dict(saved_path_dictionary)
        else:
            self.id2word, self.word2id = build_dict(
                [sentence for _, sentence in self._train_tuples],
                saved_path=saved_path_dictionary)

        glog.info('dictionary size: {}'.format(len(self.id2word)))

        self._train_tuples = self._label_to_idx(self._train_tuples, self.word2id)
        # Word may not in dev and test set.
        self._dev_tuples = self._label_to_idx(self._dev_tuples, self.word2id)
        self._test_tuples = self._label_to_idx(self._test_tuples, self.word2id)


    def _get_label_path(self, root):
        """Get the label path by root.
        
        Args:
            root: str.

        Return:
            path: str.
        """
        return osp.join(root, 'transcript', 'aishell_transcript_v0.8.txt')

    
    def _parse_label_path(self, label_path):
        """Parse labels.
        
        Args:
            label_path: str.
        
        Returns:
            label: dict, str:str. The key is voice name and the value is words.
        """
        with open(label_path, 'r', encoding='utf8') as fb:
            name2word = {}
            lines = fb.readlines()
            for line in lines:
                line = line.strip().split()
                voice_name = line[0]
                words = ''.join(line[1:])  # No need for seg words.
                name2word[voice_name] = words
        return name2word

    
    def _get_wav_paths(self, root, type='train'):
        """Get all wav file paths.
        
        Args:
            root: str.
            type: str. train or dev or test

        Returns:
            paths: list of str.
        """
        paths = []
        _train_wav_root = osp.join(root, 'wav', type)
        for root, dirs, files in os.walk(_train_wav_root):
            for name in files:
                path = osp.join(root, name)
                paths.append(path)
        return paths

    
    def _get_data_tuple(self, paths, labels, type):
        """Get the tuple of wav data path and labels.
        
        Args:
            paths: list of str.
            labels: dict. 
        """
        def _get_name(path):
            return osp.splitext(osp.split(path)[-1])[0]

        data_tuples = []
        for path in paths:
            name = _get_name(path)
            if name not in labels:
                glog.warning('type: {} ... no label for {}'.format(type, name))
                continue
            data_tuples.append((path, labels[name]))
    
        glog.info('{} tuple length ... {}'.format(type, len(data_tuples)))
        return data_tuples

    
    def _label_to_idx(self, data_tuples, word2id):
        """Change the word in data_tuple to id.

        Args:
            data_tuples: list of tuple, [(wave_path, sentences), ...]
            word2id: dict, word:id

        Returns:
            data_tuples: list of tuple, [(wave_path, sentences)]. The 
                sentences is composed with word id.
        """
        for idx, tup in enumerate(data_tuples):
            tup = list(tup)
            tup[1] = [word2id[word] for word in tup[1] if word in word2id]  # word may not in word2id 
            data_tuples[idx] = tuple(tup)
        return data_tuples


if __name__ == '__main__':
    import time
    
    root = '/share/datasets/data_aishell'
    batch_size = 32
    aishell = Aishell(root)
    t1 = time.time()
    datas = aishell.train_datas(batch_size=batch_size)
    t2 = time.time()
    print(datas)
    print(datas.shape)
    print('shuffle data time: {} with batch size: {}'.format(t2 - t1, batch_size))
