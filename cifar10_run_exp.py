import argparse
import numpy as np
import os
import json

import torch

from models import *
from run_manager import RunManager


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--gpu', help='gpu available', default='0,1,2,3')
parser.add_argument('--train', action='store_true')
parser.add_argument('--random_init', action='store_true')

parser.add_argument('--manual_seed', default=0, type=int)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--latency', type=str, default=None)

parser.add_argument('--n_epochs', type=int, default=600)
parser.add_argument('--init_lr', type=float, default=0.025)
parser.add_argument('--lr_schedule_type', type=str, default='cosine')
# lr_schedule_param

parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenet'])
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=500)
parser.add_argument('--valid_size', type=int, default=None)

parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--no_decay_keys', type=str, default='bn', choices=['None', 'bn', 'bn#bias'])

parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=10)

parser.add_argument('--n_worker', type=int, default=16)
parser.add_argument('--resize_scale', type=float, default=0.08)
parser.add_argument('--distort_color', type=str, default='strong', choices=['normal', 'strong', 'None'])
parser.add_argument('--resize_to_224', action='store_true')
parser.add_argument('--cutout', action='store_true')
parser.add_argument('--cutout_length', type=int, default=16)

""" net config """
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--bn_eps', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0)


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # os.makedirs(args.path, exist_ok=True)

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    if args.no_decay_keys == 'None':
        args.no_decay_keys = None
    run_config = Cifar10RunConfig(
        **args.__dict__
    )
    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))

    # prepare network
    net_config_path = '%s/net.config' % args.path
    if os.path.isfile(net_config_path):
        # load net from file
        from models import get_net_by_name
        net_config = json.load(open(net_config_path, 'r'))
        net = get_net_by_name(net_config['name']).build_from_config(net_config)
    else:
        raise ValueError('no net config given.[%s]' % net_config_path)

    # build run manager
    run_manager = RunManager(args.path, net, run_config, measure_latency=args.latency)
    run_manager.save_config(print_info=True)

    # load checkpoints
    init_path = '%s/init' % args.path
    if args.resume:
        run_manager.load_model()
        if args.train and run_manager.best_acc == 0:
            loss, acc1 = run_manager.validate(is_test=False, return_top5=False)
            run_manager.best_acc = acc1
    elif os.path.isfile(init_path) and not args.random_init:
        print('loading state dict......')
        if torch.cuda.is_available():
            checkpoint = torch.load(init_path)
        else:
            checkpoint = torch.load(init_path, map_location='cpu')
        assert 'state_dict' in checkpoint
        state_dict = checkpoint['state_dict']
        model_dict = run_manager.net.module.state_dict()
        model_dict.update(state_dict)
        run_manager.net.module.load_state_dict(model_dict)
        print('loaded')
    else:
        print('Random initialization')

    # train
    if args.train:
        print('Start training')
        run_manager.train(print_top5=False)
        run_manager.save_model()

    output_dict = {}
    # validate
    if run_config.valid_size:
        print('Test on validation set')
        loss, acc1 = run_manager.validate(is_test=False, return_top5=False)
        log = 'valid_loss: %f\t valid_acc1: %f' % (loss, acc1)
        run_manager.write_log(log, prefix='valid')
        output_dict = {
            **output_dict,
            'valid_loss': ' % f' % loss, 'valid_acc1': ' % f' % acc1,
            'valid_size': run_config.valid_size
        }

    # test
    print('Test on test set')
    loss, acc1 = run_manager.validate(is_test=True, return_top5=False)
    log = 'test_loss: %f\t test_acc1: %f' % (loss, acc1)
    run_manager.write_log(log, prefix='test')
    output_dict = {
        **output_dict,
        'test_loss': '%f' % loss, 'test_acc1': '%f' % acc1
    }
    json.dump(output_dict, open('%s/output' % args.path, 'w'), indent=4)