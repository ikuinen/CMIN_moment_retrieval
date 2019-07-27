import copy

import nltk
import json
from gensim.models import KeyedVectors
import h5py
import numpy as np
from torch import nn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def load_feature(filename, dataset='ActivityNet'):
    if dataset == 'ActivityNet':
        with h5py.File(filename, 'r') as fr:
            return np.asarray(fr['feature']).astype(np.float32)
    elif dataset == 'TACOS':
        return np.load(filename).astype(np.float32)
    return None


def load_json(filename):
    with open(filename, encoding='utf8') as fr:
        return json.load(fr)


def load_word2vec(filename, binary=True):
    word2vec = KeyedVectors.load_word2vec_format(filename, binary=binary)
    return word2vec


def tokenize(sentence, word2vec):
    punctuations = ['.', '?', ',', '', '(', ')']
    raw_text = sentence.lower()
    words = nltk.word_tokenize(raw_text)
    words = [word for word in words if word not in punctuations]
    return [word for word in words if word in word2vec]


def generate_anchors(dataset='ActivityNet'):
    if dataset == 'ActivityNet':
        widths = np.array([16, 32, 64, 96, 128, 160, 192])
    elif dataset == 'TACOS':
        widths = np.array([6, 18, 32])
    else:
        return None
    center = 7.5
    start = center - 0.5 * (widths - 1)
    end = center + 0.5 * (widths - 1)
    return np.stack([start, end], -1)


import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""

    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)


class StopwatchMeter(object):
    """Computes the sum/avg duration of some event in seconds"""

    def __init__(self):
        self.reset()

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.sum += delta
            self.n += n
            self.start_time = None

    def reset(self):
        self.sum = 0
        self.n = 0
        self.start_time = None

    @property
    def avg(self):
        return self.sum / self.n
