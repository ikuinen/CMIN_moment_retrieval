import collections
import logging
import os

import torch

from torch.utils.data import DataLoader

import criteria
from dataloaders.clip_loader import get_dataset
from models_.gcn_final import Model
from optimizer.adam_optimizer import AdamOptimizer
from optimizer.lr_scheduler.inverse_square_root_schedule import InverseSquareRootSchedule
from utils import load_word2vec, AverageMeter, TimeMeter
import numpy as np


class Runner:
    def __init__(self, args):
        self.num_updates = 0
        self.args = args
        self.word2vec = load_word2vec(args.word2vec_path)
        self._build_loader()
        self._build_model()
        self._build_optimizer()

    def _build_loader(self):
        args = self.args
        train = get_dataset(args.dataset, args.feature_path, args.train_data,
                            self.word2vec, args.max_num_frames, args.max_num_words,
                            args.max_num_nodes, is_training=True)
        val = get_dataset(args.dataset, args.feature_path, args.val_data,
                          self.word2vec, args.max_num_frames, args.max_num_words,
                          args.max_num_nodes, is_training=False)
        test = get_dataset(args.dataset, args.feature_path, args.test_data,
                           self.word2vec, args.max_num_frames, args.max_num_words,
                           args.max_num_nodes, is_training=False)
        self.train_loader = DataLoader(dataset=train, batch_size=self.args.batch_size, num_workers=4, shuffle=True)
        self.val_loader = DataLoader(dataset=val, batch_size=self.args.batch_size, num_workers=4,
                                     shuffle=False) if val else None
        self.test_loader = DataLoader(dataset=test, batch_size=self.args.batch_size, num_workers=4,
                                      shuffle=False) if test else None

    def _build_model(self):
        self.model = Model(self.args)
        print(self.model)
        device_ids = [0]
        # self.inference = self.model.inference
        self.model = self.model.to(torch.device('cuda:%d' % device_ids[0]))
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

    def _build_optimizer(self):
        l = list(self.model.parameters())
        self.optimizer = AdamOptimizer(self.args, list(self.model.parameters()))
        self.lr_scheduler = InverseSquareRootSchedule(self.args, self.optimizer)

    def train(self):
        if not os.path.exists(self.args.model_saved_path):
            # os.mkdir(self.args.model_saved_path)
            os.makedirs(self.args.model_saved_path)
        for epoch in range(1, self.args.max_num_epochs + 1):
            logging.info('Start Epoch {}'.format(epoch))
            self._train_one_epoch(epoch)
            path = os.path.join(self.args.model_saved_path, 'model-%d' % epoch)
            torch.save(self.model.state_dict(), path)
            logging.info('model saved to %s' % path)
            self.eval()
        logging.info('Done.')

    def _train_one_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        time_meter = TimeMeter()
        for bid, (video, video_mask, words, word_mask,
                  label, scores, scores_mask, id2pos, node_mask, adj_mat) in enumerate(self.train_loader, 1):
            self.optimizer.zero_grad()

            model_input = {
                'frames': video.cuda(),
                'frame_mask': video_mask.cuda(), 'words': words.cuda(), 'word_mask': word_mask.cuda(),
                'label': scores.cuda(), 'label_mask': scores_mask.cuda(), 'gt': label.cuda(),
                'node_pos': id2pos.cuda(), 'node_mask': node_mask.cuda(), 'adj_mat': adj_mat.cuda()
            }

            predict_boxes, loss = self.model(**model_input)
            loss = torch.mean(loss)
            self.optimizer.backward(loss)

            self.optimizer.step()
            self.num_updates += 1
            curr_lr = self.lr_scheduler.step_update(self.num_updates)

            loss_meter.update(loss.item())
            time_meter.update()

            if bid % self.args.display_n_batches == 0:
                logging.info('Epoch %d, Batch %d, loss = %.4f, lr = %.5f, %.3f seconds/batch' % (
                    epoch, bid, loss_meter.avg, curr_lr, 1.0 / time_meter.avg
                ))
                loss_meter.reset()

    def eval(self):
        data_loaders = [self.val_loader, self.test_loader]
        meters = collections.defaultdict(lambda: AverageMeter())

        self.model.eval()
        with torch.no_grad():
            for data_loader in data_loaders:
                for bid, (video, video_mask, words, word_mask,
                          label, scores, scores_mask, id2pos, node_mask, adj_mat) in enumerate(data_loader, 1):
                    self.optimizer.zero_grad()

                    model_input = {
                        'frames': video.cuda(),
                        'frame_mask': video_mask.cuda(), 'words': words.cuda(), 'word_mask': word_mask.cuda(),
                        'label': scores.cuda(), 'label_mask': scores_mask.cuda(), 'gt': label.cuda(),
                        'node_pos': id2pos.cuda(), 'node_mask': node_mask.cuda(), 'adj_mat': adj_mat.cuda()
                    }

                    predict_boxes, loss = self.model(**model_input)
                    loss = torch.mean(loss)

                    meters['loss'].update(loss.item())
                    video_mask = video_mask.cpu().numpy()
                    gt_boxes = model_input['gt'].cpu().numpy()
                    predict_boxes = np.round(predict_boxes.cpu().numpy()).astype(np.int32)
                    gt_starts, gt_ends = gt_boxes[:, 0], gt_boxes[:, 1]
                    predict_starts, predict_ends = predict_boxes[:, 0], predict_boxes[:, 1]
                    predict_starts[predict_starts < 0] = 0
                    seq_len = np.sum(video_mask, -1)
                    predict_ends[predict_ends >= seq_len] = seq_len[predict_ends >= seq_len] - 1
                    IoUs = criteria.calculate_IoU_batch((predict_starts, predict_ends),
                                                        (gt_starts, gt_ends))
                    meters['mIoU'].update(np.mean(IoUs), IoUs.shape[0])
                    for i in range(1, 10, 2):
                        meters['IoU@0.%d' % i].update(np.mean(IoUs >= (i / 10)), IoUs.shape[0])
                print('| ', end='')
                for key, value in meters.items():
                    print('{}, {:.4f}'.format(key, value.avg), end=' | ')
                    meters[key].reset()
                print()
