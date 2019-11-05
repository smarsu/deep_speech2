# Copyright 2019 smarsu. All Rights Reserved.

"""Compute cer by edit distance.

Reference:
    https://zhuanlan.zhihu.com/p/39924588
    https://leetcode.com/problems/edit-distance/
"""

import numpy as np


def min_edit_distance(predict, label):
    """Compute the min edit distant.
    
    Args:
        predict: str.
        label: str.
    """
    # print(list(predict))
    predict = [' '] + list(predict)
    label = [' '] + list(label)
    # print(predict)
    # print(label)
    dp = [[0] * len(predict) for _ in range(len(label))]
    for i in range(len(predict)):
        dp[0][i] = i
    for j in range(len(label)):
        dp[j][0] = j
    for i in range(1, len(label)):
        for j in range(1, len(predict)):
            if predict[j] == label[i]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
    
    return dp[-1][-1]


def sentence_cer(predict, label):
    """Compute the character error rate per sentence.
    
    Args:
        predict: str.
        label: str.

    Return:
        cer: float.
    """
    cer = min_edit_distance(predict, label) / len(label)
    # print(cer)
    # print()
    return cer
    

def cer(predicts, labels):
    """Compute the character error rate per corpu.
    
    Args:
        predicts: list of str.
        labels: list of str.
    
    Returns:
        cer: float.
    """
    return np.mean([sentence_cer(predict, label) 
                   for predict, label in zip(predicts, labels)])


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from datasets import datasets

    root = '/share/datasets/data_aishell'
    batch_size = 1
    aishell = datasets.Aishell(root)
    datas = aishell.train_datas(batch_size=batch_size)
    datas = datas.reshape(-1, 2)
    labels = datas[:, 1]
    print(labels)
    print(cer(labels, labels))
