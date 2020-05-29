from run_manager import *
from models.super_nets.super_progressive import ImagenetSuperProgressiveNASNets, Cifar10SuperProgressiveNASNets

SuperProgressiveNASNets = Cifar10SuperProgressiveNASNets


class ProgressiveArchSearchConfig:

    def __init__(self, target_hardware=None, reg_loss_type=None, reg_loss_params=None,
                 **kwargs):

        self.target_hardware = target_hardware
        self.reg_loss_type = reg_loss_type
        self.reg_loss_params = reg_loss_params

        print(kwargs.keys())

    @property
    def config(self):
        config = {
            'type': type(self),
        }
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config


class ArchSearchRunManager:

    def __init__(self, path, super_net, run_config: RunConfig, arch_search_config: ProgressiveArchSearchConfig):
        # init weight parameters & build weight_optimizer
        self.run_manager = RunManager(path, super_net, run_config, True)

        self.arch_search_config = arch_search_config

        self.pretrain = True
        self.pretrained_epochs = 0
        self._optimizer = None

        if isinstance(super_net, ImagenetSuperProgressiveNASNets):
            global SuperProgressiveNASNets
            SuperProgressiveNASNets = ImagenetSuperProgressiveNASNets

    @property
    def net(self):
        return self.run_manager.net.module

    def cur_mix_edge(self, idx):
        return self.net.blocks[idx].body

    def write_log(self, log_str, prefix, should_print=True, end='\n'):
        with open(os.path.join(self.run_manager.logs_path, '%s.log' % prefix), 'a') as fout:
            fout.write(log_str + end)
            fout.flush()
        if should_print:
            print(log_str)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.run_manager.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]

        if model_fname is None or not os.path.exists(model_fname):
            model_fname = '%s/checkpoint.pth.tar' % self.run_manager.save_path
            with open(latest_fname, 'w') as fout:
                fout.write(model_fname + '\n')
        if self.run_manager.out_log:
            print("=> loading checkpoint '{}'".format(model_fname))

        if torch.cuda.is_available():
            checkpoint = torch.load(model_fname)
        else:
            checkpoint = torch.load(model_fname, map_location='cpu')

        if checkpoint['mode'] == 'pretrain':
            self.pretrain = checkpoint['pretrain']
            if self.pretrain:
                self.pretrained_epochs = checkpoint['pretrained_epochs'] + 1
                self.run_manager.optimizer.load_state_dict(checkpoint['pretrained_optimizer'])
        else:   # mode == train
            self.pretrain = False
            self.run_manager.index_sel = checkpoint['index_sel']
            '''
                search layer的module保存以后,初始化super net的时候仍然有很多条候选边,无法直接调用forward函数,
                需要将super net初始化时多余的边都消除掉,因此需要在save_model的时候记录candidate op 的 index.
            '''
            for idx in self.run_manager.index_sel:
                self.cur_mix_edge(self.net.curLayer).sel_edge(idx)
                self.net.curLayer = self.net.curLayer + 1
                # self.net.blocks[].mobile_inverted_conv.sel_edge(idx)
            assert self.net.curLayer == checkpoint['cur_layer']
            self.run_manager.start_epoch = checkpoint['epoch'] + 1
            self.net.init_mix = None
            self.net.init_classifier = None
            if self.net.fix_determined:
                self.cur_mix_edge(self.net.curLayer).optimizers. \
                    load_state_dict(checkpoint['cur_layer_optimizer'])
            else:
                self._optimizer = self.run_manager.get_optimizer()
                self._optimizer.load_state_dict(checkpoint['cur_layer_optimizer'])
        self.run_manager.run_config.dataset = checkpoint['dataset']

        model_dict = self.net.state_dict()
        model_dict.update(checkpoint['state_dict'])
        self.net.load_state_dict(model_dict)
        if self.run_manager.out_log:
            print("=> loaded checkpoint '{}'".format(model_fname))

        # set new manual seed
        new_manual_seed = int(time.time())
        torch.manual_seed(new_manual_seed)
        torch.cuda.manual_seed_all(new_manual_seed)
        np.random.seed(new_manual_seed)

    """ training related methods """

    def validate(self):
        # get performances of current chosen network on validation set
        self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.run_manager.run_config.test_batch_size
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = False

        # test on validation set under train mode
        top5 = True
        if self.run_manager.run_config.dataset == 'cifar10':
            top5 = False
        valid_res = self.run_manager.validate(is_test=False, use_train_mode=True, return_top5=top5)
        # flops of chosen network
        flops = self.run_manager.net_flops()
        # measure latencies of chosen op
        if self.arch_search_config.target_hardware in [None, 'flops']:
            latency = 0
        else:
            latency, _ = self.run_manager.net_latency(
                l_type=self.arch_search_config.target_hardware, fast=False
            )

        return valid_res, flops, latency

    def train(self, re_init):     # searching process
        SuperProgressiveNASNets.MODE = 'Search'
        self.run_manager.run_config.train_loader.batch_sampler.batch_size = self.run_manager.run_config.train_batch_size
        data_loader = self.run_manager.run_config.train_loader
        nBatch = len(data_loader)
        for layer in range(self.net.curLayer, self.net.totalSearchLayer):
            assert layer == self.net.curLayer
            print('\n', '#' * 30, 'current search layer: %d' % layer, '#' * 30, '\n')
            # candidate ops
            s = '[Layer %d]Current Candidate Operations:' % layer
            for candidate_op in self.cur_mix_edge(layer).candidate_ops:
                s += ('\t' + candidate_op.module_str)
            self.run_manager.write_log(s, 'valid')
            # resume epoch which is loaded from checkpoint
            start_epoch = self.run_manager.start_epoch
            self.run_manager.start_epoch = 0
            # current layer optimizer
            if self.net.fix_determined:
                optimizer = self.cur_mix_edge(layer).optimizers
            else:
                if self._optimizer is not None:
                    optimizer = self._optimizer
                    self._optimizer = None
                else:
                    optimizer = self.run_manager.get_optimizer()
            # save trained part, abandon untrained parameters
            remove_list = []
            key_list = list(self.net.state_dict().keys())
            for i, key in enumerate(key_list):
                name = key.split('.')
                if name[0] == 'blocks' and int(name[1]) > layer:
                    remove_list = key_list[i:]
                    break
            for epoch in range(start_epoch, self.run_manager.run_config.n_epochs):
                print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
                batch_time = AverageMeter()
                data_time = AverageMeter()
                losses = [AverageMeter() for i in range(self.cur_mix_edge(layer).n_choices)]
                top1 = [AverageMeter() for i in range(self.cur_mix_edge(layer).n_choices)]
                top5 = [AverageMeter() for i in range(self.cur_mix_edge(layer).n_choices)]
                # switch to train mode
                self.run_manager.net.train()

                end = time.time()
                for i, (images, labels) in enumerate(data_loader):
                    data_time.update(time.time() - end)
                    # lr
                    lr = self.run_manager.run_config.adjust_learning_rate(
                        optimizer, epoch, batch=i, nBatch=nBatch
                    )
                    # train weight parameters if not fix_net_weights
                    images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
                    # compute output
                    output = self.run_manager.net(images)  # forward (DataParallel)
                    # compute gradient and do SGD step
                    l = []
                    init_grad = torch.tensor(1. / self.cur_mix_edge(layer).n_choices, device=self.run_manager.device)
                    optimizer.zero_grad()
                    for idx, feature in enumerate(output):
                        # loss
                        if self.run_manager.run_config.label_smoothing > 0:
                            loss = cross_entropy_with_label_smoothing(
                                feature, labels, self.run_manager.run_config.label_smoothing
                            )
                        else:
                            loss = self.run_manager.criterion(feature, labels)
                        losses[idx].update(loss.item(), images.size(0))
                        if self.net.fix_determined:
                            loss.backward(gradient=init_grad)
                        else:
                            l.append(loss)
                        # loss.backward(
                        #     gradient=init_grad, retain_graph=not self.net.fix_determined and idx != len(losses) - 1
                        # )
                        # accuracy
                        if self.run_manager.run_config.dataset == 'cifar10':
                            acc1, = accuracy(feature, labels, topk=(1,))
                            top1[idx].update(acc1[0], images.size(0))
                        else:
                            acc1, acc5 = accuracy(feature, labels, topk=(1, 5))
                            top1[idx].update(acc1[0], images.size(0))
                            top5[idx].update(acc5[0], images.size(0))
                    if not self.net.fix_determined:
                        mean_loss = sum(l) / len(l)
                        mean_loss.backward()
                    optimizer.step()  # update weight parameters
                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                    # training log
                    if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                        batch_log = 'Train [{0}][{1}/{2}]\t' \
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                    'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                    'lr {lr:.5f}'. \
                            format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time, lr=lr)
                        loss_, top1_, top5_ = '\nCandidate ops Loss(val, avg)', '\nCandidate ops Top-1 acc(val, avg)', \
                                              '\nCandidate ops Top-5 acc(val, avg)'
                        min_l, max_t, idx_l, idx_t = losses[0].avg, top1[0].avg, 0, 0
                        for ii, (_loss, _top1) in enumerate(zip(losses, top1)):
                            loss_ += ' ({0:.3f}, {1:.3f})'.format(_loss.val, _loss.avg)
                            top1_ += ' ({0:.3f}, {1:.3f})'.format(_top1.val, _top1.avg)
                            if min_l > _loss.avg:
                                min_l = _loss.avg
                                idx_l = ii
                            if max_t < _top1.avg:
                                max_t = _top1.avg
                                idx_t = ii
                        batch_log += (loss_ + top1_)
                        if self.run_manager.run_config.dataset == 'imagenet':
                            for _top5 in top5:
                                top5_ += ' ({0:.3f}, {1:.3f})'.format(_top5.val, _top5.avg)
                        else:
                            top5_ += ' None'
                        batch_log += top5_
                        batch_log += '\nThe Best Among All Candidates:idx {1} Top-1 acc {0:.3f}\tidx {3} Loss {2:.3f}\n'.\
                            format(max_t, idx_t, min_l, idx_l)
                        self.run_manager.write_log(batch_log, 'train')

                # validate
                if (epoch + 1) % self.run_manager.run_config.validation_frequency == 0:
                    if self.run_manager.run_config.dataset == 'cifar10':
                        (val_loss, val_top1, max_t1, idx_t, min_l, idx_l) = self.run_manager.\
                            train_validate(use_train_mode=True, return_top5=False,
                                           op_num=self.cur_mix_edge(layer).n_choices)
                        val_top5 = '\nCandidate ops Top-5 acc(val, avg)\tNone'
                        top5 = None
                    else:
                        (val_loss, val_top1, val_top5, max_t1, idx_t, min_l, idx_l) = self.run_manager.\
                            train_validate(use_train_mode=True, return_top5=True,
                                           op_num=self.cur_mix_edge(layer).n_choices)
                        top5 = ', '.join(['{0:.3f}'.format(e.avg) for e in top5])
                    self.run_manager.best_acc = max(self.run_manager.best_acc, max_t1)
                    val_log = 'Valid [{0}/{1}]{2}{3}{5}\n' \
                              'Train top-1 [{top1}]\ntop-5 [{top5}]\nThe Best Among ' \
                              'All Candidates:idx {7} Top-1 acc {4:.3f}\tidx {8} ' \
                              'Loss {9:.3f}\tCurrent Best Accuracy {6:.3f}\n'. \
                        format(epoch + 1, self.run_manager.run_config.n_epochs, val_loss, val_top1,
                               max_t1, val_top5, self.run_manager.best_acc, idx_t, idx_l, min_l,
                               top1=', '.join(['{0:.3f}'.format(e.avg) for e in top1]),
                               top5=top5)
                    self.run_manager.write_log(val_log, 'valid')

                # save trained part
                state_dict = self.net.state_dict()
                for key in remove_list:
                    state_dict.pop(key)
                # save model
                self.run_manager.save_model({
                    'mode': 'train',
                    'index_sel': self.run_manager.index_sel,
                    'epoch': epoch,
                    'cur_layer': layer,
                    'cur_layer_optimizer': optimizer.state_dict(),
                    'state_dict': state_dict
                })

            # print current network architecture and select one edge from mix edge
            self.cur_mix_edge(layer).accuracy = self.run_manager.val_acc(self.cur_mix_edge(layer).n_choices)
            if self.arch_search_config.target_hardware is not None:
                self.cur_mix_edge(layer).infer_latency = \
                    self.run_manager.cur_module_latency(l_type=self.arch_search_config.target_hardware, fast=False)
            self.cur_mix_edge(layer).calculate_score(self.arch_search_config.reg_loss_type,
                                                     self.arch_search_config.reg_loss_params)
            self.run_manager.index_sel.append(self.cur_mix_edge(layer).set_final_op(
                self.run_manager.run_config.determined_train or layer+1 == self.net.totalSearchLayer
            ))
            self.write_log('-' * 30 + 'Current Architecture' + '-' * 30, prefix='arch')
            for idx in range(layer + 1):    # current layer is determined
                self.write_log('%d. %s' % (idx, self.net.blocks[idx].module_str), prefix='arch')
            self.write_log(
                    'acc:[' + ', '.join(['{0:.3f}'.format(e) for e in self.cur_mix_edge(layer).accuracy]) + ']',
                    prefix='arch')
            if len(self.cur_mix_edge(layer).infer_latency):
                self.write_log(
                    'infer_latency(ms):['+', '.join(['{0:.3f}'.format(e) for e in self.cur_mix_edge(layer).infer_latency])+']',
                    prefix='arch')
            else:
                self.write_log('infer_latency: []', prefix='arch')
            self.write_log(
                'score:[' + ', '.join(['{0:.3f}'.format(e) for e in self.cur_mix_edge(layer).rank_list]) + ']',
                prefix='arch')
            self.write_log('arch: [' + ', '.join([str(e) for e in self.run_manager.index_sel]) + ']', prefix='arch')
            self.write_log('-' * 60, prefix='arch')

            if self.run_manager.run_config.determined_train:    # train determined part
                if re_init == 'last_determined':
                    self.net.cur_block_mixlayer.init_model()
                elif re_init == 'determined_part':
                    self.net.init_model(self.run_manager.run_config.model_init)
                elif re_init == 'no_reinit':
                    pass
                else:
                    raise ValueError('Not implemented.')
                self.determined_train(re_init, layer+1 != self.net.totalSearchLayer)     # remove unused modules

            self.net.curLayer = self.net.curLayer + 1

        SuperProgressiveNASNets.MODE = 'Finish'
        # convert to normal network according to architecture parameters
        self.net.cpu().convert_to_normal_net()
        print('Model total params: %.2fM' % (count_parameters(self.net) / 1e6))
        os.makedirs(os.path.join(self.run_manager.path, 'learned_net'), exist_ok=True)
        json.dump(super(SuperProgressiveNASNets, self.net).config,
                  open(os.path.join(self.run_manager.path, 'learned_net/net.config'), 'w'), indent=4)
        json.dump(
            self.run_manager.run_config.config,
            open(os.path.join(self.run_manager.path, 'learned_net/run.config'), 'w'), indent=4,
        )
        torch.save(
            {'state_dict': self.net.state_dict(), 'dataset': self.run_manager.run_config.dataset},
            os.path.join(self.run_manager.path, 'learned_net/init')
        )
        print('net architecture: [' + ', '.join([str(e) for e in self.run_manager.index_sel]) + '].')

    def init_train(self, pretrain_epochs, pretrained_batch):
        SuperProgressiveNASNets.MODE = 'Pretrain'

        lr_max = 0.05
        self.run_manager.run_config.train_loader.batch_sampler.batch_size = pretrained_batch
        data_loader = self.run_manager.run_config.train_loader
        nBatch = len(data_loader)
        T_total = pretrain_epochs * nBatch

        # save pretrained part
        remove_list = []
        key_list = list(self.net.state_dict().keys())
        for i, key in enumerate(key_list):
            name = key.split('.')
            if name[0] == 'blocks' and int(name[1]) == self.net.curLayer:
                remove_list = key_list[i:]
                break

        for epoch in range(self.pretrained_epochs, pretrain_epochs):
            print('\n', '-' * 30, 'Pretrain epoch: %d' % (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            # switch to train mode
            self.run_manager.net.train()

            end = time.time()
            for i, (images, labels) in enumerate(data_loader):
                data_time.update(time.time() - end)
                # lr
                T_cur = epoch * nBatch + i
                pretrain_lr = 0.5 * lr_max * (1 + math.cos(math.pi * T_cur / T_total))
                for param_group in self.run_manager.optimizer.param_groups:
                    param_group['lr'] = pretrain_lr
                images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
                # compute output
                output = self.run_manager.net(images)  # forward (DataParallel)
                # loss
                if self.run_manager.run_config.label_smoothing > 0:
                    loss = cross_entropy_with_label_smoothing(
                        output, labels, self.run_manager.run_config.label_smoothing
                    )
                else:
                    loss = self.run_manager.criterion(output, labels)
                # accuracy
                if self.run_manager.run_config.dataset == 'cifar10':
                    acc1, = accuracy(output, labels, topk=(1, ))
                    top1.update(acc1[0], images.size(0))
                else:
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))

                losses.update(loss, images.size(0))
                # compute gradient and do SGD step
                self.run_manager.optimizer.zero_grad()
                loss.backward()
                self.run_manager.optimizer.step()  # update weight parameters
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                    batch_log = 'Pre Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'lr {lr:.5f}\tTop-1 acc {top1.val:.3f} ({top1.avg:.3f})' . \
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses, top1=top1, lr=pretrain_lr)
                    if self.run_manager.run_config.dataset == 'imagenet':
                        batch_log = batch_log + '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'. \
                            format(top5=top5)
                    self.run_manager.write_log(batch_log, 'pretrain')

            if self.run_manager.run_config.dataset == 'cifar10':
                (val_loss, val_top1) = self.run_manager.validate(is_test=False, use_train_mode=True, return_top5=False)
                val_top5 = None
                top5_avg = None
            else:
                (val_loss, val_top1, val_top5) = self.run_manager.validate(is_test=False, use_train_mode=True,
                                                                           return_top5=True)
                val_top5 = '{0.3f}'.format(val_top5)
                top5_avg = '{0.3f}'.format(top5.avg)
            val_log = 'Pretrain Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f}\ttop-5 acc {4}\t' \
                      'Train top-1 {top1.avg:.3f}\ttop-5 {top5}'.\
                format(epoch + 1, pretrain_epochs, val_loss, val_top1, val_top5, top1=top1, top5=top5_avg)
            self.run_manager.write_log(val_log, 'valid')
            self.pretrain = epoch + 1 < pretrain_epochs

            # remove unnecessary module and optimizer that hold their reference if pretrain finished
            if not self.pretrain:
                self.net.init_classifier = None
                self.net.init_mix = None
                self.run_manager.optimizer = None

            # save pretrained part
            state_dict = self.net.state_dict()
            for key in remove_list:
                state_dict.pop(key)

            checkpoint = {
                'mode': 'pretrain',
                'state_dict': state_dict,
                'pretrain': self.pretrain,
            }
            if self.pretrain:
                checkpoint['pretrained_epochs'] = epoch
                checkpoint['pretrained_optimizer'] = self.run_manager.optimizer.state_dict()
            self.run_manager.save_model(checkpoint, model_name='pretrain.pth.tar')
        # torch.cuda.empty_cache()

    def determined_train(self, re_init, remove_unused_modules):
        SuperProgressiveNASNets.MODE = 'Determined_train'

        # optimizer = self.run_manager.get_optimizer()
        opt_param = {} if self.run_manager.run_config.opt_param is None else self.run_manager.run_config.opt_param
        momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
        optimizer = torch.optim.SGD(list(self.net.weight_parameters()) + list(self.net.cur_block_mixlayer.parameters()),
                                    0.05, momentum=momentum, nesterov=nesterov,
                                    weight_decay=self.run_manager.run_config.weight_decay)
        print_top5 = (self.run_manager.run_config.dataset == 'imagenet')
        self.run_manager.run_config.train_loader.batch_sampler.batch_size = \
            self.run_manager.run_config.determined_train_batch
        lr_max = 0.05
        data_loader = self.run_manager.run_config.train_loader
        nBatch = len(data_loader)
        T_total = self.run_manager.run_config.determined_train_epoch * nBatch

        train_epoch = self.run_manager.run_config.determined_train_epoch
        # if re_init == 'last_determined':
        #     train_epoch = self.run_manager.run_config.determined_train_epoch
        # else:
        #     train_epoch = int(
        #         self.run_manager.run_config.determined_train_epoch * (1 + self.net.curLayer / self.net.totalSearchLayer)
        #     )
        """
        train_epoch can grow with the depth of determined part or other strategy.
        E.g. according to the last layer type of determined part, train_epoch has 
        fixed epoch w.r.t that, or just simple set to a fixed value no matter what
        there is.
        """

        for epoch in range(train_epoch):
            print('\n', '-' * 30, 'Determined Train Epoch: %d' % (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            # switch to train mode
            self.run_manager.net.train()

            end = time.time()
            for i, (images, labels) in enumerate(data_loader):
                data_time.update(time.time() - end)
                # lr
                T_cur = epoch * nBatch + i
                lr = 0.5 * lr_max * (1 + math.cos(math.pi * T_cur / T_total))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
                # compute output
                output = self.run_manager.net(images)  # forward (DataParallel)
                # loss
                if self.run_manager.run_config.label_smoothing > 0:
                    loss = cross_entropy_with_label_smoothing(
                        output, labels, self.run_manager.run_config.label_smoothing
                    )
                else:
                    loss = self.run_manager.criterion(output, labels)
                # measure accuracy and record loss
                if print_top5:
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))
                else:
                    acc1, = accuracy(output, labels, topk=(1,))
                    top1.update(acc1[0], images.size(0))
                losses.update(loss, images.size(0))
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # update weight parameters
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                    batch_log = 'Determined Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'lr {lr:.5f}\tTop-1 acc {top1.val:.3f} ({top1.avg:.3f})'. \
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses, top1=top1, lr=lr)
                    if print_top5:
                        batch_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
                    self.run_manager.write_log(batch_log, 'determined_train')
            if print_top5:
                (val_loss, val_top1, val_top5) = self.run_manager.validate(is_test=False, use_train_mode=True,
                                                                           return_top5=True)
                val_top5 = '{0.3f}'.format(val_top5)
                top5_avg = '{0.3f}'.format(top5.avg)
            else:
                (val_loss, val_top1) = self.run_manager.validate(is_test=False, use_train_mode=True, return_top5=False)
                val_top5 = None
                top5_avg = None
            val_log = 'Determined Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f}\ttop-5 acc {4}\t' \
                      'Train top-1 {top1.avg:.3f}\ttop-5 {top5}'. \
                format(epoch + 1, train_epoch, val_loss, val_top1, val_top5, top1=top1, top5=top5_avg)
            self.run_manager.write_log(val_log, 'valid')

        if remove_unused_modules:
            self.net.cur_block_mixlayer.remove_unused()
        self.run_manager.run_config.train_loader.batch_sampler.batch_size = self.run_manager.run_config.train_batch_size
        SuperProgressiveNASNets.MODE = 'Search'
