import time
import json
from datetime import timedelta
import numpy as np
import copy

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from utils import *
# from models.normal_nets.progressive_nets import ProgressiveNASNets
from modules.mix_op import MixedEdge


class RunConfig:

    def __init__(self, determined_train, determined_train_epoch, determined_train_batch,
                 n_epochs, init_lr, lr_schedule_type, lr_schedule_param, dataset, train_batch_size,
                 test_batch_size, valid_size, opt_type, opt_param, weight_decay,
                 label_smoothing, no_decay_keys, low_lr_parts, model_init, init_div_groups,
                 validation_frequency, print_frequency):
        self.determined_train = determined_train
        self.determined_train_epoch = determined_train_epoch
        self.determined_train_batch = determined_train_batch
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys
        self.low_lr_parts = low_lr_parts

        self.model_init = model_init
        self.init_div_groups = init_div_groups
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency

        self._data_provider = None
        self._train_iter, self._valid_iter, self._test_iter = None, None, None

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    """ learning rate """

    def _calc_learning_rate(self, epoch, batch=0, nBatch=None):
        if self.lr_schedule_type == 'cosine':
            T_total = self.n_epochs * nBatch
            T_cur = epoch * nBatch + batch
            lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))
        else:
            raise ValueError('do not support: %s' % self.lr_schedule_type)
        return lr

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self._calc_learning_rate(epoch, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    """ data provider """

    @property
    def data_config(self):
        raise NotImplementedError

    @property
    def data_provider(self):
        if self._data_provider is None:
            if self.dataset == 'imagenet':
                from data_providers.imagenet import ImagenetDataProvider
                self._data_provider = ImagenetDataProvider(**self.data_config)
            elif self.dataset == 'cifar10':
                from data_providers.cifar10 import Cifar10DataProvider
                self._data_provider = Cifar10DataProvider(**self.data_config)
            else:
                raise ValueError('do not support: %s' % self.dataset)
        return self._data_provider

    @data_provider.setter
    def data_provider(self, val):
        self._data_provider = val

    @property
    def train_loader(self):
        return self.data_provider.train

    @property
    def valid_loader(self):
        return self.data_provider.valid

    @property
    def test_loader(self):
        return self.data_provider.test

    @property
    def train_next_batch(self):
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            data = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            data = next(self._train_iter)
        return data

    @property
    def valid_next_batch(self):
        if self._valid_iter is None:
            self._valid_iter = iter(self.valid_loader)
        try:
            data = next(self._valid_iter)
        except StopIteration:
            self._valid_iter = iter(self.valid_loader)
            data = next(self._valid_iter)
        return data

    @property
    def test_next_batch(self):
        if self._test_iter is None:
            self._test_iter = iter(self.test_loader)
        try:
            data = next(self._test_iter)
        except StopIteration:
            self._test_iter = iter(self.test_loader)
            data = next(self._test_iter)
        return data

    """ optimizer """

    def build_undetermined_optimizer(self, net):
        if self.opt_type == 'sgd':
            opt_param = {} if self.opt_param is None else self.opt_param
            momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
            for i in range(net.curLayer, net.totalSearchLayer):
                net.blocks[i].body.build_optimizers(self.init_lr, momentum=momentum, nesterov=nesterov,
                                                    weight_decay=self.weight_decay, no_decay_keys=self.no_decay_keys)
        else:
            raise NotImplementedError

    def build_undetermined_optimizer1(self, net):
        if self.opt_type == 'sgd':
            opt_param = {} if self.opt_param is None else self.opt_param
            momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
            # param_list = [net.first_conv.parameters()]
            if self.low_lr_parts is not None:
                if self.low_lr_parts == 'determined':
                    optimizer = torch.optim.SGD([
                        {'params': net.weight_parameters(), 'lr': self.init_lr / 2},
                        {'params': net.cur_block_mixlayer.parameters(), 'lr': self.init_lr}
                    ], self.init_lr, momentum=momentum, nesterov=nesterov, weight_decay=self.weight_decay)
                else:
                    raise NotImplementedError
            else:
                optimizer = torch.optim.SGD(list(net.weight_parameters()) + list(net.cur_block_mixlayer.parameters()),
                                            self.init_lr, momentum=momentum, nesterov=nesterov,
                                            weight_decay=self.weight_decay)
        else:
            raise NotImplementedError
        return optimizer

    def build_determined_optimizer(self, net_params):
        if self.opt_type == 'sgd':
            opt_param = {} if self.opt_param is None else self.opt_param
            momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
            if self.no_decay_keys:
                optimizer = torch.optim.SGD([
                    {'params': net_params[0], 'weight_decay': self.weight_decay},
                    {'params': net_params[1], 'weight_decay': 0},
                ], lr=self.init_lr, momentum=momentum, nesterov=nesterov)
            else:
                optimizer = torch.optim.SGD(net_params, self.init_lr, momentum=momentum, nesterov=nesterov,
                                            weight_decay=self.weight_decay)
        else:
            raise NotImplementedError
        return optimizer


class RunManager:

    def __init__(self, path, net, run_config: RunConfig, out_log=True, measure_latency=None):
        self.path = path
        self.net = net
        self.run_config = run_config
        self.out_log = out_log

        self._logs_path, self._save_path = None, None
        self.best_acc = 0
        self.start_epoch = 0
        self.index_sel = []     # selected mixed edge id

        # initialize model (default)
        self.net.init_model(run_config.model_init, run_config.init_div_groups)

        # a copy of net on cpu for latency estimation & mobile latency model
        self.net_on_cpu_for_latency = copy.deepcopy(self.net).cpu()
        self.latency_estimator = LatencyEstimator()

        # move network to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.net = torch.nn.DataParallel(self.net)
            self.net.to(self.device)
            cudnn.benchmark = True
        else:
            raise ValueError
            # self.device = torch.device('cpu')

        # net info
        self.print_net_info(measure_latency)

        self.criterion = nn.CrossEntropyLoss()

        # set optimizer for each layer if determined layers' weight is not fixed
        if self.net.module.fix_determined:
            self.run_config.build_undetermined_optimizer(self.net.module)

        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split('#')
            self.optimizer = self.run_config.build_determined_optimizer([
                self.net.module.get_parameters(keys, mode='exclude'),  # parameters with weight decay
                self.net.module.get_parameters(keys, mode='include'),  # parameters without weight decay
            ])
        else:
            self.optimizer = self.run_config.build_determined_optimizer(self.net.module.weight_parameters())

    """ save path and log path """

    @property
    def save_path(self):
        if self._save_path is None:
            save_path = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return self._logs_path

    def get_optimizer(self):
        # assert not self.net.module.fix_determined
        # net = self.net.module
        return self.run_config.build_undetermined_optimizer1(self.net.module)

    """ net info """

    # noinspection PyUnresolvedReferences
    def net_flops(self):
        data_shape = [1] + list(self.run_config.data_provider.data_shape)

        if isinstance(self.net, nn.DataParallel):
            net = self.net.module
        else:
            net = self.net
        input_var = torch.zeros(data_shape, device=self.device)
        net.eval()
        with torch.no_grad():
            flop, _ = net.get_flops(input_var)
        net.train()
        return flop

    def cur_module_latency(self, l_type='gpu1', fast=True, given_net=None):
        latency = []
        if 'gpu' in l_type:
            l_type, batch_size = l_type[:3], int(l_type[3:])
        else:
            batch_size = 1

        if given_net is not None:
            net = given_net
        else:
            net = self.net.module

        fsize = self.run_config.data_provider.data_shape[2]    # find out current layer's in and out size
        idskip = 0
        out_fz = fsize // net.first_conv.stride
        for idx in range(net.curLayer + 1):
            fsize = out_fz  # previous layer out size
            conv = net.blocks[idx].body
            shortcut = net.blocks[idx].shortcut
            if isinstance(conv, MixedEdge):
                conv = conv.candidate_ops[0]
            if shortcut is None:
                idskip = 0
            else:
                idskip = 1
            out_fz = fsize // conv.stride  # current layer out size

        data_shape = [batch_size] + [net.cur_block.body.candidate_ops[0].in_channels] + [fsize, fsize]
        out_data_shape = [batch_size] + [net.cur_block.body.candidate_ops[0].real_outchannels] + [out_fz, out_fz]

        self.write_log('measure layer [{0}] latency, in data shape [{1}], out data shape [{2}]\n'.\
                       format(net.curLayer, data_shape, out_data_shape), prefix='valid')

        if l_type == 'mobile':
            for conv in net.cur_block.body.candidate_ops:
                block_latency = self.latency_estimator.predict(
                    'expanded_conv', [fsize, fsize, conv.in_channels], [out_fz, out_fz, conv.out_channels],
                    expand=conv.expand_ratio, kernel=conv.kernel_size, stride=conv.stride, idskip=idskip
                )
                latency.append(block_latency)
            return latency
        elif l_type == 'cpu':
            if fast:
                n_warmup = 1
                n_sample = 2
            else:
                n_warmup = 10
                n_sample = 100
            times, sigma = 3, [2, 2, 2]
            net = self.net_on_cpu_for_latency
            net.curLayer = self.net.module.curLayer
            images = torch.zeros(data_shape, device=torch.device('cpu'))
        elif l_type == 'gpu':
            if fast:
                n_warmup = 50
                n_sample = 100
            else:
                n_warmup = 50
                n_sample = 1000
            times, sigma = 3, [2, 1, 2]
            images = torch.zeros(data_shape, device=self.device)
        else:
            raise NotImplementedError

        net.eval()
        with torch.no_grad():
            for module in net.cur_block.body.candidate_ops:
                measured_latency = []
                for i in range(n_warmup + n_sample):
                    start_time = time.time()
                    tmp = module(images)
                    if shortcut:
                        tmp1 = shortcut(images)
                        if tmp1.shape[1] != tmp.shape[1]:
                            padding = torch.zeros(
                                batch_size, tmp.shape[1] - tmp1.shape[1], tmp.shape[2], tmp.shape[3],
                                device=tmp.device, requires_grad=False
                            )
                            tmp1 = torch.cat((tmp1, padding), 1)
                        tmp += tmp1
                    used_time = (time.time() - start_time) * 1e3  # ms
                    if i >= n_warmup:
                        measured_latency.append(used_time)
                _, avg_latency, _ = filter_data(measured_latency, times=times, sigma=sigma)
                latency.append(avg_latency)
        net.train()
        return latency

    # noinspection PyUnresolvedReferences
    def net_latency(self, l_type='gpu1', fast=True, given_net=None):
        if 'gpu' in l_type:
            l_type, batch_size = l_type[:3], int(l_type[3:])
        else:
            batch_size = 1

        data_shape = [batch_size] + list(self.run_config.data_provider.data_shape)

        if given_net is not None:
            net = given_net
        else:
            net = self.net.module

        if l_type == 'mobile':
            predicted_latency = 0
            try:
                assert isinstance(net, ProxylessNASNets)
                # first conv
                predicted_latency += self.latency_estimator.predict(
                    'Conv', [224, 224, 3], [112, 112, net.first_conv.out_channels]
                )
                # feature mix layer
                predicted_latency += self.latency_estimator.predict(
                    'Conv_1', [7, 7, net.feature_mix_layer.in_channels], [7, 7, net.feature_mix_layer.out_channels]
                )
                # classifier
                predicted_latency += self.latency_estimator.predict(
                    'Logits', [7, 7, net.classifier.in_features], [net.classifier.out_features]  # 1000
                )
                # blocks
                fsize = 112
                for block in net.blocks:
                    mb_conv = block.body
                    shortcut = block.shortcut
                    if isinstance(mb_conv, MixedEdge):
                        mb_conv = mb_conv.candidate_ops[0]
                    if isinstance(shortcut, MixedEdge):
                        shortcut = shortcut.candidate_ops[0]

                    if mb_conv.is_zero_layer():
                        continue
                    if shortcut is None or shortcut.is_zero_layer():
                        idskip = 0
                    else:
                        idskip = 1
                    out_fz = fsize // mb_conv.stride
                    block_latency = self.latency_estimator.predict(
                        'expanded_conv', [fsize, fsize, mb_conv.in_channels], [out_fz, out_fz, mb_conv.out_channels],
                        expand=mb_conv.expand_ratio, kernel=mb_conv.kernel_size, stride=mb_conv.stride, idskip=idskip
                    )
                    predicted_latency += block_latency
                    fsize = out_fz
            except Exception:
                predicted_latency = 200
                print('fail to predict the mobile latency')
            return predicted_latency, None
        elif l_type == 'cpu':
            if fast:
                n_warmup = 1
                n_sample = 2
            else:
                n_warmup = 10
                n_sample = 100
            net = self.net_on_cpu_for_latency
            images = torch.zeros(data_shape, device=torch.device('cpu'))
        elif l_type == 'gpu':
            if fast:
                n_warmup = 5
                n_sample = 10
            else:
                n_warmup = 50
                n_sample = 100
            images = torch.zeros(data_shape, device=self.device)
        else:
            raise NotImplementedError

        measured_latency = {'warmup': [], 'sample': []}
        net.eval()
        with torch.no_grad():
            for i in range(n_warmup + n_sample):
                start_time = time.time()
                net(images)
                used_time = (time.time() - start_time) * 1e3  # ms
                if i >= n_warmup:
                    measured_latency['sample'].append(used_time)
                else:
                    measured_latency['warmup'].append(used_time)
        net.train()
        return sum(measured_latency['sample']) / n_sample, measured_latency

    def print_net_info(self, measure_latency=None):
        # network architecture
        if self.out_log:
            print(self.net)

        # parameters
        if isinstance(self.net, nn.DataParallel):
            total_params = count_parameters(self.net.module)
        else:
            total_params = count_parameters(self.net)
        if self.out_log:
            print('Total training params: %.2fM' % (total_params / 1e6))
        net_info = {
            'param': '%.2fM' % (total_params / 1e6),
        }

        # flops
        flops = self.net_flops()
        if self.out_log:
            print('Total FLOPs: %.1fM' % (flops / 1e6))
        net_info['Total flops'] = '%.1fM' % (flops / 1e6)

        # latency
        latency_types = [] if measure_latency is None else measure_latency.split('#')
        for l_type in latency_types:
            latency, measured_latency = self.net_latency(l_type, fast=False, given_net=None)
            if self.out_log:
                print('Estimated %s latency: %.3fms' % (l_type, latency))
            net_info['%s latency' % l_type] = {
                'val': latency,
                'hist': measured_latency
            }
        with open('%s/net_info.txt' % self.logs_path, 'w') as fout:
            fout.write(json.dumps(net_info, indent=4) + '\n')

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.net.module.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'

        checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
        # checkpoint['fix_determined'] = self.net.module.fix_determined
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]
        # noinspection PyBroadException
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = '%s/checkpoint.pth.tar' % self.save_path
                with open(latest_fname, 'w') as fout:
                    fout.write(model_fname + '\n')
            if self.out_log:
                print("=> loading checkpoint '{}'".format(model_fname))

            if torch.cuda.is_available():
                checkpoint = torch.load(model_fname)
            else:
                checkpoint = torch.load(model_fname, map_location='cpu')

            self.net.module.load_state_dict(checkpoint['state_dict'])
            # set new manual seed
            new_manual_seed = int(time.time())
            torch.manual_seed(new_manual_seed)
            torch.cuda.manual_seed_all(new_manual_seed)
            np.random.seed(new_manual_seed)

            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                self.best_acc = checkpoint['best_acc']
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            if self.out_log:
                print("=> loaded checkpoint '{}'".format(model_fname))
        except Exception:
            if self.out_log:
                print('fail to load checkpoint from %s' % self.save_path)

    def save_config(self, print_info=True):
        """ dump run_config and net_config to the model_folder """
        os.makedirs(self.path, exist_ok=True)
        net_save_path = os.path.join(self.path, 'net.config')
        json.dump(self.net.module.config, open(net_save_path, 'w'), indent=4)
        if print_info:
            print('Network configs dump to %s' % net_save_path)

        run_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4)
        if print_info:
            print('Run configs dump to %s' % run_save_path)

    """ train and test """

    def write_log(self, log_str, prefix, should_print=True):
        """ prefix: valid, train, test """
        if prefix in ['valid', 'test']:
            with open(os.path.join(self.logs_path, 'valid_console.txt'), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if prefix in ['valid', 'test', 'train', 'pretrain', 'determined_train']:
            with open(os.path.join(self.logs_path, 'train_console.txt'), 'a') as fout:
                if prefix in ['valid', 'test']:
                    fout.write('=' * 50)
                fout.write(log_str + '\n')
                fout.flush()
        if should_print:
            print(log_str)

    def val_acc(self, op_num):
        data_loader = self.run_config.valid_loader
        net = self.net
        net.eval()      # model.training = False?
        # losses = [AverageMeter() for i in range(op_num)]
        top1 = [AverageMeter() for i in range(op_num)]
        # top5 = [AverageMeter() for i in range(op_num)]

        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                output = net(images)

                for idx, feature in enumerate(output):
                    acc1, = accuracy(feature, labels, topk=(1, ))
                    # loss = self.criterion(feature, labels)
                    # acc1, acc5 = accuracy(feature, labels, topk=(1, 5))
                    top1[idx].update(acc1[0], images.size(0))
                    # top5[idx].update(acc5[0], images.size(0))
                    # losses[idx].update(loss, images.size(0))
        net.train()
        return [acc.avg.item() for acc in top1]

    def train_validate(self, net=None, use_train_mode=False, return_top5=False, op_num=0):
        data_loader = self.run_config.valid_loader

        if net is None:
            net = self.net

        if use_train_mode:
            net.train()
        else:
            net.eval()
        batch_time = AverageMeter()
        losses = [AverageMeter() for i in range(op_num)]
        top1 = [AverageMeter() for i in range(op_num)]
        top5 = [AverageMeter() for i in range(op_num)]

        end = time.time()
        # noinspection PyUnresolvedReferences
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                output = net(images)

                for idx, feature in enumerate(output):
                    loss = self.criterion(feature, labels)
                    # measure accuracy and record loss
                    if return_top5:
                        acc1, acc5 = accuracy(feature, labels, topk=(1, 5))
                        top1[idx].update(acc1[0], images.size(0))
                        top5[idx].update(acc5[0], images.size(0))
                    else:
                        acc1, = accuracy(feature, labels, topk=(1, ))
                        top1[idx].update(acc1[0], images.size(0))
                    losses[idx].update(loss, images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.run_config.print_frequency == 0 or i + 1 == len(data_loader):
                    prefix = 'Valid'
                    test_log = prefix + ': [{0}/{1}]\t'\
                                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.\
                        format(i, len(data_loader) - 1, batch_time=batch_time)
                    loss_, top1_, top5_ = '\nCandidate ops Loss(val, avg)', '\nCandidate ops Top-1 acc(val, avg)', \
                                          '\nCandidate ops Top-5 acc(val, avg)'
                    max_t1, min_l, idx_t, idx_l = top1[0].avg, losses[0].avg, 0, 0
                    for ii, (_loss, _top1) in enumerate(zip(losses, top1)):
                        loss_ += ' ({0:.4f}, {1:.4f})'.format(_loss.val, _loss.avg)
                        top1_ += ' ({0:.3f}, {1:.3f})'.format(_top1.val, _top1.avg)
                        if max_t1 < _top1.avg:
                            max_t1 = _top1.avg
                            idx_t = ii
                        if min_l > _loss.avg:
                            min_l = _loss.avg
                            idx_l = ii
                    test_log += (loss_ + top1_)
                    if return_top5:
                        for _top5 in top5:
                            top5_ += ' ({0:.3f}, {1:.3f})'.format(_top5.val, _top5.avg)
                        test_log += top5_
                    test_log += '\nThe Best Among All Candidates: idx {0} Top-1 acc {1:.3f}\tidx {2} Loss {3:.3f}\n'.\
                        format(idx_t, max_t1, idx_l, min_l)
                    # self.write_log(test_log, 'valid')
                    print(test_log)
        if return_top5:
            return loss_, top1_, top5_, max_t1, idx_t, min_l, idx_l
        else:
            return loss_, top1_, max_t1, idx_t, min_l, idx_l

    def validate(self, is_test=True, net=None, use_train_mode=False, return_top5=False):
        if is_test:
            data_loader = self.run_config.test_loader
        else:
            data_loader = self.run_config.valid_loader

        if net is None:
            net = self.net

        if use_train_mode:
            net.train()
        else:
            net.eval()
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        # noinspection PyUnresolvedReferences
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                output = net(images)
                loss = self.criterion(output, labels)
                # measure accuracy and record loss
                if return_top5:
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))
                else:
                    acc1, = accuracy(output, labels, topk=(1, ))    # 这个逗号太重要了！！！！！！
                    top1.update(acc1[0], images.size(0))
                losses.update(loss, images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.run_config.print_frequency == 0 or i + 1 == len(data_loader):
                    test_log = ': [{0}/{1}]\t' \
                               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                               'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                               'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'.\
                        format(i, len(data_loader) - 1, batch_time=batch_time, losses=losses, top1=top1)
                    if return_top5:
                        test_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
                    if is_test:
                        test_log = 'Test' + test_log
                    else:
                        test_log = 'Valid' + test_log
                    print(test_log)
        if return_top5:
            return losses.avg, top1.avg, top5.avg
        else:
            return losses.avg, top1.avg

    def train_one_epoch(self, adjust_lr_func, train_log_func, return_top5=False):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.net.train()

        end = time.time()
        for i, (images, labels) in enumerate(self.run_config.train_loader):
            data_time.update(time.time() - end)
            new_lr = adjust_lr_func(i)
            images, labels = images.to(self.device), labels.to(self.device)

            # compute output
            output = self.net(images)
            if self.run_config.label_smoothing > 0:
                loss = cross_entropy_with_label_smoothing(output, labels, self.run_config.label_smoothing)
            else:
                loss = self.criterion(output, labels)

            # measure accuracy and record loss
            losses.update(loss, images.size(0))
            if return_top5:
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
            else:
                acc1, = accuracy(output, labels, topk=(1,))
                top1.update(acc1[0], images.size(0))

            # compute gradient and do SGD step
            self.net.zero_grad()  # or self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.run_config.print_frequency == 0 or i + 1 == len(self.run_config.train_loader):
                batch_log = train_log_func(i, batch_time, data_time, losses, top1, top5, new_lr)
                self.write_log(batch_log, 'train')
        return top1, top5

    def train(self, print_top5=False):
        nBatch = len(self.run_config.train_loader)

        def train_log_func(epoch_, i, batch_time, data_time, losses, top1, top5, lr):
            batch_log = 'Train [{0}][{1}/{2}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                        'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'. \
                format(epoch_ + 1, i, nBatch - 1,
                       batch_time=batch_time, data_time=data_time, losses=losses, top1=top1)
            if print_top5:
                batch_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
            batch_log += '\tlr {lr:.5f}'.format(lr=lr)
            return batch_log

        for epoch in range(self.start_epoch, self.run_config.n_epochs):
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')

            end = time.time()
            train_top1, train_top5 = self.train_one_epoch(
                lambda i: self.run_config.adjust_learning_rate(self.optimizer, epoch, i, nBatch),
                lambda i, batch_time, data_time, losses, top1, top5, new_lr:
                train_log_func(epoch, i, batch_time, data_time, losses, top1, top5, new_lr),
                return_top5=print_top5)
            time_per_epoch = time.time() - end
            seconds_left = int((self.run_config.n_epochs - epoch - 1) * time_per_epoch)
            print('Time per epoch: %s, Est. complete in: %s' % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if (epoch + 1) % self.run_config.validation_frequency == 0:
                if print_top5:
                    val_loss, val_acc, val_acc5 = self.validate(is_test=False, return_top5=True)
                else:
                    val_loss, val_acc = self.validate(is_test=False, return_top5=False)
                is_best = val_acc > self.best_acc
                self.best_acc = max(self.best_acc, val_acc)
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})'.\
                    format(epoch + 1, self.run_config.n_epochs, val_loss, val_acc, self.best_acc)
                if print_top5:
                    val_log += '\ttop-5 acc {0:.3f}\tTrain top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}'.\
                        format(val_acc5, top1=train_top1, top5=train_top5)
                else:
                    val_log += '\tTrain top-1 {top1.avg:.3f}'.format(top1=train_top1)
                self.write_log(val_log, 'valid')
            else:
                is_best = False

            self.save_model({
                'epoch': epoch,
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.net.module.state_dict(),
            }, is_best=is_best)
