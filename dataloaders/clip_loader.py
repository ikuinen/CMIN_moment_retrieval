import numpy as np
import criteria
from datasets.ActivityNet import ActivityNetGCN
from datasets.TACOS import TACOSGCN
from utils import load_json, generate_anchors


def get_dataset(dataset, feature_path, data_path, word2vec, max_num_frames, max_num_words, max_num_nodes,
                is_training=True):
    if dataset == 'ActivityNet':
        return ClipDataset1(feature_path, data_path, word2vec, max_num_frames, max_num_words, max_num_nodes,
                            is_training)
    elif dataset == 'TACOS':
        return ClipDataset2(feature_path, data_path, word2vec, max_num_frames, max_num_words, max_num_nodes,
                            is_training)
    return None


class ClipDataset1(ActivityNetGCN):
    def __init__(self, feature_path, data_path, word2vec,
                 max_num_frames, max_num_words, max_num_nodes, is_training=True):
        data = load_json(data_path)
        super().__init__(feature_path, data, word2vec, is_training)
        self.max_num_frames = max_num_frames
        self.max_num_words = max_num_words
        self.max_num_nodes = max_num_nodes
        self.anchors = generate_anchors(dataset='ActivityNet')
        widths = (self.anchors[:, 1] - self.anchors[:, 0] + 1)  # [num_anchors]
        centers = np.arange(0, max_num_frames)  # [video_len]
        start = np.expand_dims(centers, 1) - 0.5 * (np.expand_dims(widths, 0) - 1)
        end = np.expand_dims(centers, 1) + 0.5 * (np.expand_dims(widths, 0) - 1)
        self.proposals = np.stack([start, end], -1)  # [video_len, num_anchors, 2]

    # coarse sampling
    def __getitem__(self, index):
        video, words, label, id2pos, adj_mat = super().__getitem__(index)

        ori_words_len = words.shape[0]
        # word padding
        if ori_words_len < self.max_num_words:
            word_mask = np.zeros([self.max_num_words], np.uint8)
            word_mask[range(ori_words_len)] = 1
            words = np.pad(words, ((0, self.max_num_words - ori_words_len), (0, 0)), mode='constant')
        else:
            word_mask = np.ones([self.max_num_words], np.uint8)
            words = words[:self.max_num_words]

        # video sampling
        ori_video_len = video.shape[0]
        video_mask = np.ones([self.max_num_frames], np.uint8)
        index = np.linspace(start=0, stop=ori_video_len - 1, num=self.max_num_frames).astype(np.int32)
        new_video = []
        for i in range(len(index) - 1):
            start = index[i]
            end = index[i + 1]
            if start == end or start + 1 == end:
                new_video.append(video[start])
            else:
                new_video.append(np.mean(video[start: end], 0))
        new_video.append(video[-1])
        video = np.stack(new_video, 0)

        # label recomputing
        # print(index, label)
        label[0] = min(np.where(index >= label[0])[0])
        if label[1] == video.shape[0] - 1:
            label[1] = self.max_num_frames - 1
        else:
            label[1] = max(np.where(index <= label[1])[0])
        if label[1] < label[0]:
            label[0] = label[1]

        assert len(id2pos) == adj_mat.shape[0] == ori_words_len

        # some words have been cut out
        true_index = id2pos < self.max_num_words
        id2pos = id2pos[true_index]
        adj_mat = adj_mat[true_index]
        adj_mat = adj_mat[:, true_index]

        # node padding
        if id2pos.shape[0] < self.max_num_nodes:
            node_mask = np.zeros([self.max_num_nodes], np.uint8)
            node_mask[range(id2pos.shape[0])] = 1
            id2pos = np.pad(id2pos, (0, self.max_num_nodes - id2pos.shape[0]), mode='constant')
            adj_mat = np.pad(adj_mat,
                             ((0, self.max_num_nodes - adj_mat.shape[0]),
                              (0, self.max_num_nodes - adj_mat.shape[1])),
                             mode='constant')
        else:
            node_mask = np.ones([self.max_num_nodes], np.uint8)
            id2pos = id2pos[:self.max_num_nodes]
            adj_mat = adj_mat[:self.max_num_nodes, :self.max_num_nodes]

        # scores computing
        proposals = np.reshape(self.proposals, [-1, 2])
        illegal = np.logical_or(proposals[:, 0] < 0, proposals[:, 1] >= self.max_num_frames)
        label1 = np.repeat(np.expand_dims(label, 0), proposals.shape[0], 0)
        IoUs = criteria.calculate_IoU_batch((proposals[:, 0], proposals[:, 1]),
                                            (label1[:, 0], label1[:, 1]))
        IoUs[illegal] = 0.0  # [video_len * num_anchors]
        max_IoU = np.max(IoUs)
        if max_IoU == 0.0:
            print(illegal)
            print(label)
            print(proposals[illegal])
            print(proposals[1 - illegal])
            # print(IoUs)
            # print(label, max_IoU)
            exit(1)
        IoUs[IoUs < 0.3 * max_IoU] = 0.0
        IoUs = IoUs / max_IoU
        scores = IoUs.astype(np.float32)
        scores_mask = (1 - illegal).astype(np.uint8)
        return video, video_mask, words, word_mask, label, \
               scores, scores_mask, \
               id2pos, node_mask, adj_mat


class ClipDataset2(TACOSGCN):
    def __init__(self, feature_path, data_path, word2vec,
                 max_num_frames, max_num_words, max_num_nodes, is_training=True):
        data = load_json(data_path)
        super().__init__(feature_path, data, word2vec, is_training)
        self.max_num_frames = max_num_frames
        self.max_num_words = max_num_words
        self.max_num_nodes = max_num_nodes
        self.anchors = generate_anchors(dataset='TACOS')
        widths = (self.anchors[:, 1] - self.anchors[:, 0] + 1)  # [num_anchors]
        centers = np.arange(0, max_num_frames)  # [video_len]
        start = np.expand_dims(centers, 1) - 0.5 * (np.expand_dims(widths, 0) - 1)
        end = np.expand_dims(centers, 1) + 0.5 * (np.expand_dims(widths, 0) - 1)
        self.proposals = np.stack([start, end], -1)  # [video_len, num_anchors, 2]

    # coarse sampling
    def __getitem__(self, index):
        video, words, label, id2pos, adj_mat = super().__getitem__(index)

        ori_words_len = words.shape[0]
        # word padding
        if ori_words_len < self.max_num_words:
            word_mask = np.zeros([self.max_num_words], np.uint8)
            word_mask[range(ori_words_len)] = 1
            words = np.pad(words, ((0, self.max_num_words - ori_words_len), (0, 0)), mode='constant')
        else:
            word_mask = np.ones([self.max_num_words], np.uint8)
            words = words[:self.max_num_words]

        # video sampling
        ori_video_len = video.shape[0]
        video_mask = np.ones([self.max_num_frames], np.uint8)
        index = np.linspace(start=0, stop=ori_video_len - 1, num=self.max_num_frames).astype(np.int32)
        new_video = []
        for i in range(len(index) - 1):
            start = index[i]
            end = index[i + 1]
            if start == end or start + 1 == end:
                new_video.append(video[start])
            else:
                new_video.append(np.mean(video[start: end], 0))
        new_video.append(video[-1])
        video = np.stack(new_video, 0)

        # label recomputing
        # print(index, label)
        label[0] = min(np.where(index >= label[0])[0])
        if label[1] == video.shape[0] - 1:
            label[1] = self.max_num_frames - 1
        else:
            label[1] = max(np.where(index <= label[1])[0])
        if label[1] < label[0]:
            label[0] = label[1]

        assert len(id2pos) == adj_mat.shape[0] == ori_words_len

        # some words have been cut out
        true_index = id2pos < self.max_num_words
        id2pos = id2pos[true_index]
        adj_mat = adj_mat[true_index]
        adj_mat = adj_mat[:, true_index]

        # node padding
        if id2pos.shape[0] < self.max_num_nodes:
            node_mask = np.zeros([self.max_num_nodes], np.uint8)
            node_mask[range(id2pos.shape[0])] = 1
            id2pos = np.pad(id2pos, (0, self.max_num_nodes - id2pos.shape[0]), mode='constant')
            adj_mat = np.pad(adj_mat,
                             ((0, self.max_num_nodes - adj_mat.shape[0]),
                              (0, self.max_num_nodes - adj_mat.shape[1])),
                             mode='constant')
        else:
            node_mask = np.ones([self.max_num_nodes], np.uint8)
            id2pos = id2pos[:self.max_num_nodes]
            adj_mat = adj_mat[:self.max_num_nodes, :self.max_num_nodes]

        # scores computing
        proposals = np.reshape(self.proposals, [-1, 2])
        illegal = np.logical_or(proposals[:, 0] < 0, proposals[:, 1] >= self.max_num_frames)
        label1 = np.repeat(np.expand_dims(label, 0), proposals.shape[0], 0)
        IoUs = criteria.calculate_IoU_batch((proposals[:, 0], proposals[:, 1]),
                                            (label1[:, 0], label1[:, 1]))
        IoUs[illegal] = 0.0  # [video_len * num_anchors]
        max_IoU = np.max(IoUs)
        if max_IoU == 0.0:
            print(illegal)
            print(label)
            print(proposals[illegal])
            print(proposals[1 - illegal])
            # print(IoUs)
            # print(label, max_IoU)
            exit(1)
        IoUs[IoUs < 0.3 * max_IoU] = 0.0
        IoUs = IoUs / max_IoU
        scores = IoUs.astype(np.float32)
        scores_mask = (1 - illegal).astype(np.uint8)
        return video, video_mask, words, word_mask, label, \
               scores, scores_mask, \
               id2pos, node_mask, adj_mat
