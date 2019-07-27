import argparse
import logging


# activity-net: epoch 9
# tacos:
from optimizer.lr_scheduler import LR_SCHEDULER_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--dataset', choices=['ActivityNet', 'TACOS'], default='ActivityNet',
                        help='')
    parser.add_argument('--train-data', type=str,
                        default=None,
                        help='')
    parser.add_argument('--val-data', type=str, default=None,
                        help='')
    parser.add_argument('--test-data', type=str, default=None,
                        help='')
    parser.add_argument('--word2vec-path', type=str, default='glove_model.bin',
                        help='')
    parser.add_argument('--feature-path', type=str, default='data/activity-c3d',
                        help='')
    parser.add_argument('--model-saved-path', type=str, default='results/model_%s',
                        help='')  # 2, 4, 3 'results/model_main/model-best'
    parser.add_argument('--max-num-words', type=int, default=20,
                        help='')
    parser.add_argument('--max-num-nodes', type=int, default=20,
                        help='')
    parser.add_argument('--max-num-frames', type=int, default=200,
                        help='')
    parser.add_argument('--d-model', type=int, default=512,
                        help='')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='')
    parser.add_argument('--word-dim', type=int, default=300,
                        help='')
    parser.add_argument('--frame-dim', type=int, default=500,
                        help='')
    parser.add_argument('--num-gcn-layers', type=int, default=2,
                        help='')
    parser.add_argument('--num-attn-layers', type=int, default=2,
                        help='')
    parser.add_argument('--display-n-batches', type=int, default=50,
                        help='')
    parser.add_argument('--max-num-epochs', type=int, default=20,
                        help='')
    parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                       help='weight decay')
    parser.add_argument('--lr-scheduler', default='inverse_sqrt',
                 choices=LR_SCHEDULER_REGISTRY.keys(),
                 help='Learning Rate Scheduler')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    from optim.lr_scheduler.inverse_square_root_schedule import InverseSquareRootSchedule
    InverseSquareRootSchedule.add_args(parser)
    from optimizer.adam_optimizer import AdamOptimizer
    AdamOptimizer.add_args(parser)
    return parser.parse_args()


def main(args):
    print(args)
    from runners.runner_final import Runner
    runner = Runner(args)
    if args.train:
        runner.train()
    if args.evaluate:
        runner.eval()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args = parse_args()
    main(args)
