# Copyright 2019 smarsu. All Rights Reserved.

"""Implement of decode ctc output."""

import numpy as np


def ctc_greedy_decoder_per_batch(input, black_index=0):
    """
    Args:
        input: ndarray, shape [batch, time, num_classes]. The batchsize 
            should be 1.
        black_index: int.

    Returns:
        decoded_input: list, shape [time].
    """
    _, time, num_classes = input.shape
    input = input.reshape(time, num_classes)
    input = np.argmax(input, -1)
    assert input.shape == (time, )

    decoded_input = [input[0]]
    for word in input[1:]:
        if word == decoded_input[-1]:
            continue
        else:
            decoded_input.append(word)

    decoded_input = [word for word in decoded_input if word != black_index]
    return decoded_input


def ctc_greedy_decoder(inputs, sequence_length, black_index=0):
    """
    Args:
        inputs: ndarray, shape [batch, time, num_classes]
        sequence_length: list of int, shape [batch]

    Returns:
        decoded_inputs: list of list. The inter list have different length.
    """
    decoded_inputs = []
    for i in range(len(inputs)):
        decoded_input = ctc_greedy_decoder_per_batch(
            inputs[i:i+1, :sequence_length[i]], 
            black_index=black_index)
        decoded_inputs.append(decoded_input)
    
    return decoded_inputs
