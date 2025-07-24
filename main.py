import os
import sys
import traceback
import argparse
import yaml

import torch
import numpy as np
import random

from train import Trainer
from utils import Report

class YamlAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        yaml_dict = yaml.safe_load(values)
        setattr(namespace, self.dest, yaml_dict)

def get_parser():
    parser = argparse.ArgumentParser(description='Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition')
    parser.add_argument('--work-dir', default='./work_dir', help='the work folder for storing results')
    parser.add_argument('--config', default='./config/test.yaml', help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-iter', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-iter', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=0, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-epoch', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--val-epoch', type=int, default=5, help='the interval for evaluating models (#iteration)')

    # feeder
    parser.add_argument('--feeder', default='feeders.feeder', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=4, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', action=YamlAction, default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--val-feeder-args', action=YamlAction, default=dict(), help='the arguments of data loader for validation')
    parser.add_argument('--test-feeder-args', action=YamlAction, default=dict(), help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args', action=YamlAction, default=dict(), help='the arguments of model')

    # optim
    parser.add_argument('--gpu', type=int, default=0, help='the index of GPUs for training or testing')
    parser.add_argument('--optimizer', default='AdamW', help='type of optimizer')
    parser.add_argument('--lr-scheduler', default='cosine', help='type of learning rate scheduler')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--num-iter', type=int, default=1, help='the number of total iteration')
    parser.add_argument('--num-epoch', type=int, default=1, help='the number of total iteration')
    parser.add_argument('--num-warmup', type=int, default=1, help='the number of warmup iteration')
    parser.add_argument('--batch-size', type=int, default=16, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=1, help='test batch size')
    parser.add_argument('--mixed-precision', type=str, default=None, choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--pretrained-model-name-or-path', type=str, default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--accelerator-path', type=str, default='stabilityai/stable-diffusion-2-1-base')

    # zero-shot learning (zsl)
    parser.add_argument('--unseen_label', type=int, default=5, help='the number of unseen classes')
    parser.add_argument('--unseen_label_path', type=str, default='./data/label_splits/ntu60/ru5.npy', help='list of unseen classes')
    parser.add_argument('--prediction-type', type=str, default='eplison', help='eplison or sample or v_prediction')
    parser.add_argument('--d-weight', type=float, default=1.0, help='the weight of diffusion loss')
    parser.add_argument('--t-weight', type=float, default=1.0, help='the weight of triplet loss')
    parser.add_argument('--num-noise', type=int, default=3, help='the number of inference noise')
    parser.add_argument('--margin', type=float, default=1.0, help='the margin for triplet loss')

    # denoising model
    parser.add_argument('--in-channels', type=int, default=256)
    parser.add_argument('--hidden-size', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num-heads', type=int, default=12)

    # test
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--idx-inference-step', type=int, default=25)
    return parser


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_data(args):
    Feeder = import_class(args.feeder)
    data_loader = dict()
    data_loader['train'] = torch.utils.data.DataLoader(
        dataset=Feeder(**args.train_feeder_args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        drop_last=True)
    data_loader['val'] = torch.utils.data.DataLoader(
        dataset=Feeder(**args.val_feeder_args),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        drop_last=False)
    data_loader['test'] = torch.utils.data.DataLoader(
        dataset=Feeder(**args.test_feeder_args),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        drop_last=False)
    return data_loader


def train(args):
    global_step = 0
    train_log = Report(args.work_dir, type='train')
    val_log = Report(args.work_dir, type='val')
    data_loader = load_data(args)

    trainer = Trainer(args=args, data_loader=data_loader)

    best_zsl_acc = 0
    best_zsl_epoch = 0
    last_epoch = 0
    total_epoch = args.num_iter // len(data_loader['train']) + 1
    for epoch in range(last_epoch, total_epoch):
        train_log.write(f'========= Epoch {epoch + 1} of {total_epoch} =========')
        global_step = trainer.train(train_log, global_step)

        # Test ZSL (Zero-Shot Learning) Accuracy
        zsl_test_acc = trainer.test()
        val_log.write(f'ZSL Test Acc: {zsl_test_acc:.6f}\tEpoch: {epoch + 1}')
        if zsl_test_acc > best_zsl_acc:
            best_zsl_acc = zsl_test_acc
            best_zsl_epoch = epoch + 1
            trainer.save_best_model()
            val_log.write(f'Best Test Acc: {best_zsl_acc:.6f}\tBest Epoch: {best_zsl_epoch}')


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_args = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_args.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_args)

    args = parser.parse_args()
    init_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    train(args)