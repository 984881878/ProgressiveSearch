import torch.nn.functional as F
from modules.layers import *
import numpy as np
from scipy.stats import norm


def build_candidate_ops(candidate_ops, in_channels, out_channels, stride):
    if candidate_ops is None:
        raise ValueError('please specify a candidate set')

    name2ops = {
        'Zero': lambda in_C, out_C, S: ZeroLayer(in_C, out_C, stride=S),
    }

    name2ops.update({
        '3x3_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 1),
        '3x3_MBConv2': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 2),
        '3x3_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 3),
        '3x3_MBConv4': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 4),
        '3x3_MBConv5': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 5),
        '3x3_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 6),
        #######################################################################################
        '5x5_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 1),
        '5x5_MBConv2': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 2),
        '5x5_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 3),
        '5x5_MBConv4': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 4),
        '5x5_MBConv5': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 5),
        '5x5_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 6),
        #######################################################################################
        '7x7_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 1),
        '7x7_MBConv2': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 2),
        '7x7_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 3),
        '7x7_MBConv4': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 4),
        '7x7_MBConv5': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 5),
        '7x7_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 6),
    })

    name2ops.update({
        '3x3_SEMBConv1': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 3, S, 1, se_reduction_ratio=16),
        '3x3_SEMBConv2': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 3, S, 2, se_reduction_ratio=16),
        '3x3_SEMBConv3': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 3, S, 3, se_reduction_ratio=16),
        '3x3_SEMBConv4': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 3, S, 4, se_reduction_ratio=16),
        '3x3_SEMBConv5': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 3, S, 5, se_reduction_ratio=16),
        '3x3_SEMBConv6': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 3, S, 6, se_reduction_ratio=16),
        #############################################################################################
        '5x5_SEMBConv1': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 5, S, 1, se_reduction_ratio=16),
        '5x5_SEMBConv2': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 5, S, 2, se_reduction_ratio=16),
        '5x5_SEMBConv3': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 5, S, 3, se_reduction_ratio=16),
        '5x5_SEMBConv4': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 5, S, 4, se_reduction_ratio=16),
        '5x5_SEMBConv5': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 5, S, 5, se_reduction_ratio=16),
        '5x5_SEMBConv6': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 5, S, 6, se_reduction_ratio=16),
        #############################################################################################
        '7x7_SEMBConv1': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 7, S, 1, se_reduction_ratio=16),
        '7x7_SEMBConv2': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 7, S, 2, se_reduction_ratio=16),
        '7x7_SEMBConv3': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 7, S, 3, se_reduction_ratio=16),
        '7x7_SEMBConv4': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 7, S, 4, se_reduction_ratio=16),
        '7x7_SEMBConv5': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 7, S, 5, se_reduction_ratio=16),
        '7x7_SEMBConv6': lambda in_C, out_C, S: SEMBConv(in_C, out_C, 7, S, 6, se_reduction_ratio=16),
    })

    name2ops.update({
        'Bottleneck_3x3Conv_Expand0': lambda in_C, out_C, S: Bottleneck(in_C, out_C, 3, S, 0, se_reduction_ratio=16),
        'Bottleneck_3x3Conv_Expand1': lambda in_C, out_C, S: Bottleneck(in_C, out_C, 3, S, 1, se_reduction_ratio=16),
        'Bottleneck_3x3Conv_Expand2': lambda in_C, out_C, S: Bottleneck(in_C, out_C, 3, S, 2, se_reduction_ratio=16),
        'Bottleneck_3x3Conv_Expand3': lambda in_C, out_C, S: Bottleneck(in_C, out_C, 3, S, 3, se_reduction_ratio=16),
        #############################################################################################################
        'Bottleneck_5x5Conv_Expand0': lambda in_C, out_C, S: Bottleneck(in_C, out_C, 5, S, 0, se_reduction_ratio=16),
        'Bottleneck_5x5Conv_Expand1': lambda in_C, out_C, S: Bottleneck(in_C, out_C, 5, S, 1, se_reduction_ratio=16),
        'Bottleneck_5x5Conv_Expand2': lambda in_C, out_C, S: Bottleneck(in_C, out_C, 5, S, 2, se_reduction_ratio=16),
        'Bottleneck_5x5Conv_Expand3': lambda in_C, out_C, S: Bottleneck(in_C, out_C, 5, S, 3, se_reduction_ratio=16),
        #############################################################################################################
        'Bottleneck_7x7Conv_Expand0': lambda in_C, out_C, S: Bottleneck(in_C, out_C, 7, S, 0, se_reduction_ratio=16),
        'Bottleneck_7x7Conv_Expand1': lambda in_C, out_C, S: Bottleneck(in_C, out_C, 7, S, 1, se_reduction_ratio=16),
        'Bottleneck_7x7Conv_Expand2': lambda in_C, out_C, S: Bottleneck(in_C, out_C, 7, S, 2, se_reduction_ratio=16),
        'Bottleneck_7x7Conv_Expand3': lambda in_C, out_C, S: Bottleneck(in_C, out_C, 7, S, 3, se_reduction_ratio=16),
        #############################################################################################################
        'Bottleneck_11x11Conv_Expand0': lambda in_C, out_C, S: Bottleneck(in_C, out_C, 11, S, 0, se_reduction_ratio=16),
        'Bottleneck_11x11Conv_Expand1': lambda in_C, out_C, S: Bottleneck(in_C, out_C, 11, S, 1, se_reduction_ratio=16),
        'Bottleneck_11x11Conv_Expand2': lambda in_C, out_C, S: Bottleneck(in_C, out_C, 11, S, 2, se_reduction_ratio=16),
        'Bottleneck_11x11Conv_Expand3': lambda in_C, out_C, S: Bottleneck(in_C, out_C, 11, S, 3, se_reduction_ratio=16),

    })

    name2ops.update({
        'BasicBlock_3x3Conv_Expand0': lambda in_C, out_C, S: BasicBlock(in_C, out_C, 3, S, 0),
        'BasicBlock_3x3Conv_Expand1': lambda in_C, out_C, S: BasicBlock(in_C, out_C, 3, S, 1),
        'BasicBlock_3x3Conv_Expand2': lambda in_C, out_C, S: BasicBlock(in_C, out_C, 3, S, 2),
        #############################################################################################
        'BasicBlock_5x5Conv_Expand0': lambda in_C, out_C, S: BasicBlock(in_C, out_C, 5, S, 0),
        'BasicBlock_5x5Conv_Expand1': lambda in_C, out_C, S: BasicBlock(in_C, out_C, 5, S, 1),
        'BasicBlock_5x5Conv_Expand2': lambda in_C, out_C, S: BasicBlock(in_C, out_C, 5, S, 2),
        #############################################################################################
        'BasicBlock_7x7Conv_Expand0': lambda in_C, out_C, S: BasicBlock(in_C, out_C, 7, S, 0),
        'BasicBlock_7x7Conv_Expand1': lambda in_C, out_C, S: BasicBlock(in_C, out_C, 7, S, 1),
        'BasicBlock_7x7Conv_Expand2': lambda in_C, out_C, S: BasicBlock(in_C, out_C, 7, S, 2),
        #############################################################################################
        'BasicBlock_11x11Conv_Expand0': lambda in_C, out_C, S: BasicBlock(in_C, out_C, 11, S, 0),
        'BasicBlock_11x11Conv_Expand1': lambda in_C, out_C, S: BasicBlock(in_C, out_C, 11, S, 1),
        'BasicBlock_11x11Conv_Expand2': lambda in_C, out_C, S: BasicBlock(in_C, out_C, 11, S, 2),
    })

    op_list = [
        name2ops[name](in_channels, out_channels, stride) for name in candidate_ops if name != 'Zero'
    ]

    if 'Zero' in candidate_ops:
        out_channels = op_list[0].real_outchannels
        op_list.append(name2ops['Zero'](in_channels, out_channels, stride))

    return op_list


class MixedEdge(MyModule):

    def __init__(self, candidate_ops, n_classes, enable_mix=True, share_mix_layer=False,
                 share_classifier=True, dropout_rate=0):
        super(MixedEdge, self).__init__()

        self.enable_mix = enable_mix
        self.share_mix_layer = share_mix_layer
        self.share_classifier = share_classifier

        self.candidate_ops = nn.ModuleList(candidate_ops)
        self.candidate_ops_mixed_layer = None
        self.candidate_ops_classifier = None
        in_channels = self.candidate_ops[0].real_outchannels
        if enable_mix:
            if share_mix_layer:
                self.candidate_ops_mixed_layer = ConvLayer(
                    in_channels, in_channels * 4, kernel_size=1, use_bn=True, act_func='relu6',
                    ops_order='weight_bn_act',
                )
            else:
                self.candidate_ops_mixed_layer = nn.ModuleList([ConvLayer(
                    in_channels, in_channels * 4, kernel_size=1, use_bn=True, act_func='relu6',
                    ops_order='weight_bn_act',
                ) for i in range(len(self.candidate_ops))])
            in_channels = in_channels * 4

        if share_classifier:
            self.candidate_ops_classifier = LinearLayer(in_channels, n_classes, dropout_rate=dropout_rate)
        else:
            self.candidate_ops_classifier = nn.ModuleList([LinearLayer(in_channels, n_classes,
                                                                       dropout_rate=dropout_rate)
                                                           for i in range(len(self.candidate_ops))])

        self.rank_list = []
        self.accuracy = []
        self.infer_latency = []

        self._optimizer = None     # contain optimizers from each edge
        self._residual = True     # used in serching process
        self._mode = 'cifar10'
        self._shortcut = None

    def forward(self, x):
        for i, module in enumerate(self.candidate_ops):
            res = module(x)
            if self.n_choices > 1:
                if self.residual:
                    if self.mode == 'cifar10':
                        if self.shortcut == 'identity':
                            skip_x = x
                        elif self.shortcut == 'avgpool':
                            skip_x = F.avg_pool2d(x, 2, 2)
                        else:
                            raise ValueError('not implemented')
                        batch_size, residual_channel, featuremap_size = res.size()[0], res.size()[1], res.size()[2:4]
                        shortcut_channel = skip_x.size()[1]
                        if residual_channel != shortcut_channel:
                            padding = torch.autograd.Variable(
                                torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel,
                                                       featuremap_size[0], featuremap_size[1]).fill_(0))
                            res += torch.cat((skip_x, padding), 1)
                        else:
                            res += skip_x
                    elif self.mode == 'imagenet':
                        res = res + x
                    else:
                        raise ValueError('not implemented')
                if self.enable_mix:
                    if self.share_mix_layer:
                        res = self.candidate_ops_mixed_layer(res)
                    else:
                        res = self.candidate_ops_mixed_layer[i](res)
                # res = F.adaptive_avg_pool2d(res, 1)
                # res = res.view(res.size(0), -1)
                batch_size, num_channels, H, W = res.size()
                res = res.view(batch_size, num_channels, -1).mean(dim=2)
                if self.share_classifier:
                    res = self.candidate_ops_classifier(res)
                else:
                    res = self.candidate_ops_classifier[i](res)
            yield res
        # return x

    def calculate_score(self, score_type, param):
        self.rank_list = []
        if len(self.infer_latency):
            assert len(self.accuracy) == len(self.infer_latency)

            acc = np.array(self.accuracy)
            acc = (acc - acc.mean()) / acc.std()    # standardize accuracy
            acc = [norm.cdf(itm) for itm in acc]

            latency = np.array(self.infer_latency)
            latency = (latency - latency.mean()) / latency.std()    # standardize latency
            latency = [1 - norm.cdf(itm) for itm in latency]

            if score_type == 'mul':
                for _acc, _latency in zip(acc, latency):
                    self.rank_list.append(_acc * _latency * 100)
            elif score_type == 'add':
                w_acc = param.get('w_acc', 0.7)
                w_latency = param.get('w_latency', 0.3)
                for _acc, _latency in zip(acc, latency):
                    self.rank_list.append((w_acc * _acc + w_latency * _latency) * 100)
            else:
                raise NotImplementedError
        else:
            self.rank_list = self.accuracy

    def set_final_op(self, keep_unused=False):
        assert len(self.rank_list) == len(self.candidate_ops)
        target = self.rank_list[-1]
        target_i = len(self.rank_list) - 1
        for i, score in enumerate(self.rank_list):
            if target < score:
                target = score
                target_i = i

        self._optimizer = None
        if not keep_unused:
            self.remove_unused()
        else:
            if isinstance(self.candidate_ops_classifier, nn.ModuleList):
                self.candidate_ops_classifier = self.candidate_ops_classifier[target_i]
            if isinstance(self.candidate_ops_mixed_layer, nn.ModuleList):
                self.candidate_ops_mixed_layer = self.candidate_ops_mixed_layer[target_i]
        self.candidate_ops = nn.ModuleList([self.candidate_ops[target_i]])
        # torch.cuda.empty_cache()
        # gc.collect()
        return target_i

    def remove_unused(self):
        # self._optimizer = None
        self.candidate_ops_classifier = None
        self.candidate_ops_mixed_layer = None

    def sel_edge(self, idx):
        self._optimizer = None
        self.candidate_ops_classifier = None
        self.candidate_ops_mixed_layer = None
        self.candidate_ops = nn.ModuleList([self.candidate_ops[idx]])
        # torch.cuda.empty_cache()
        # gc.collect()

    @property
    def n_choices(self):
        return len(self.candidate_ops)

    @property
    def residual(self):
        return self._residual

    @residual.setter
    def residual(self, val):
        self._residual = val

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, val):
        self._mode = val

    @property
    def shortcut(self):
        return self._shortcut

    @shortcut.setter
    def shortcut(self, val):
        self._shortcut = val

    @property
    def module_str(self):
        assert len(self.candidate_ops) == 1
        return '%s' % self.candidate_ops[0].module_str
        # return '%s' % self._modules.values().module_str

    @property
    def config(self):
        raise ValueError('not needed')

    @property
    def optimizers(self):
        if self._optimizer is None:
            raise ValueError('not initialized')
        return self._optimizer

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    def get_flops(self, x):
        flops = 0
        for module in self._modules.values():
            delta_flop, _ = module.get_flops(x)
            flops += delta_flop
        return flops, self.forward(x)

    def build_optimizers(self, init_lr, momentum, nesterov, weight_decay, no_decay_keys):
        if no_decay_keys:
            keys = no_decay_keys.split('#')
            decay_part, no_decay_part = [], []
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag:
                    no_decay_part.append(param)
                else:
                    decay_part.append(param)
            optimizer = torch.optim.SGD([
                {'params': decay_part, 'weight_decay': weight_decay},
                {'params': no_decay_part, 'weight_decay': 0},
            ], lr=init_lr, momentum=momentum, nesterov=nesterov)
        else:
            optimizer = torch.optim.SGD(self.parameters(), init_lr, momentum=momentum, nesterov=nesterov,
                                        weight_decay=weight_decay)
        self._optimizer = optimizer
        # return self._optimizer

    def init_model(self, model_init='he_fout', init_div_groups=False):
        assert self.n_choices == 1  # only called when candidate operations has been determined
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
