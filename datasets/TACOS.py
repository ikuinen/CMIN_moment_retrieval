import os
import numpy as np
from torch.utils.data import Dataset

from utils import load_feature, tokenize


class TACOS(Dataset):
    def __init__(self, feature_path, data, word2vec, is_training=True):
        self.data = data
        self.feature_path = feature_path
        self.is_training = is_training
        self.word2vec = word2vec

    def __getitem__(self, index):
        vid, duration, timestamps, sentence = self.data[index]
        feats = load_feature(os.path.join(self.feature_path, '%s.npy' % vid[:-4]), dataset='TACOS')
        fps = feats.shape[0] / duration

        start_frame = int(fps * timestamps[0])
        end_frame = int(fps * timestamps[1])
        if end_frame >= feats.shape[0]:
            end_frame = feats.shape[0] - 1
        if start_frame > end_frame:
            start_frame = end_frame
        assert start_frame <= end_frame
        assert 0 <= start_frame < feats.shape[0]
        assert 0 <= end_frame < feats.shape[0]
        label = np.asarray([start_frame, end_frame]).astype(np.int32)

        words = tokenize(sentence, self.word2vec)
        words_vec = np.asarray([self.word2vec[word] for word in words])
        words_vec = words_vec.astype(np.float32)

        return feats, words_vec, label

    def __len__(self):
        return len(self.data)


class TACOSGCN(Dataset):
    def __init__(self, feature_path, data, word2vec, is_training=True):
        self.data = data
        self.feature_path = feature_path
        self.is_training = is_training
        self.word2vec = word2vec

    def __getitem__(self, index):
        vid, duration, timestamps, sentence, words, id2pos, adj_mat = self.data[index]
        feats = load_feature(os.path.join(self.feature_path, '%s.npy' % vid[:-4]), dataset='TACOS')
        fps = feats.shape[0] / duration
        adj_mat = np.asarray(adj_mat)
        start_frame = int(fps * timestamps[0])
        end_frame = int(fps * timestamps[1])
        if end_frame >= feats.shape[0]:
            end_frame = feats.shape[0] - 1
        if start_frame > end_frame:
            start_frame = end_frame
        assert start_frame <= end_frame
        assert 0 <= start_frame < feats.shape[0]
        assert 0 <= end_frame < feats.shape[0]
        label = np.asarray([start_frame, end_frame]).astype(np.int32)

        words_vec = np.asarray([self.word2vec[word] for word in words])
        words_vec = words_vec.astype(np.float32)

        id2pos = np.asarray(id2pos).astype(np.int64)
        return feats, words_vec, label, id2pos, adj_mat.astype(np.int32)

    def __len__(self):
        return len(self.data)
