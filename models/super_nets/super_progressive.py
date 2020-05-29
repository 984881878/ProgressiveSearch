from queue import Queue
import copy

from modules.mix_op import *
from models.normal_nets.progressive_nets import *
from utils import LatencyEstimator


class ImagenetSuperProgressiveNASNets(ImagenetProgressiveNASNets):
    MODE = 'Pretrain'  # Pretrain  Search  Finish

    def __init__(self, width_stages, n_cell_stages, conv_candidates, stride_stages,
                 n_classes=10, width_mult=1, bn_param=(0.1, 1e-3), dropout_rate=0,
                 enable_mix=True, share_mix_layer=False, share_classifier=False,
                 enable_init_mix=True, fix_determined=False):

        input_channel = make_divisible(32 * width_mult, 8)
        first_cell_width = make_divisible(16 * width_mult, 8)   # 这里的16可以调整 24,32都可以尝试
        for i in range(len(width_stages)):
            width_stages[i] = make_divisible(width_stages[i] * width_mult, 8)

        # first conv layer
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act'
        )

        first_block_conv = build_candidate_ops(['3x3_MBConv1'], input_channel, first_cell_width, 1)[0]
        first_block = MobileInvertedResidualBlock(first_block_conv, None)
        input_channel = first_cell_width

        # blocks
        blocks = [first_block]
        # blocks = []
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                if stride == 1 and input_channel == width:
                    modified_conv_candidates = conv_candidates + ['Zero']
                else:
                    modified_conv_candidates = conv_candidates
                conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                    modified_conv_candidates, input_channel, width, stride,
                ), n_classes=n_classes, enable_mix=enable_mix, share_mix_layer=share_mix_layer,
                    share_classifier=share_classifier, dropout_rate=dropout_rate
                )
                # shortcut
                if stride == 1 and input_channel == width:
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None
                inverted_residual_block = MobileInvertedResidualBlock(conv_op, shortcut)
                blocks.append(inverted_residual_block)
                input_channel = width

        # feature mix layer
        last_channel = make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        feature_mix_layer = ConvLayer(
            input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act',
        )

        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)
        super(ImagenetSuperProgressiveNASNets, self).__init__(first_conv, blocks, feature_mix_layer, classifier,
                                                              enable_init_mix, fix_determined)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    def forward(self, x):
        if ImagenetSuperProgressiveNASNets.MODE == 'Pretrain':
            x = self.first_conv(x)
            for i in range(self.curLayer):
                x = self.blocks[i](x)
            if self.init_mix is not None:
                x = self.init_mix(x)
            # x = F.adaptive_avg_pool2d(x, 1)
            # x = x.view(x.size(0), -1)
            batch_size, num_channels, H, W = x.size()
            x = x.view(batch_size, num_channels, -1).mean(dim=2)
            x = self.init_classifier(x)
        elif ImagenetSuperProgressiveNASNets.MODE == 'Search':
            if self.fix_determined:
                with torch.no_grad():
                    x = self.first_conv(x)
                    for i in range(self.curLayer):
                        x = self.blocks[i](x)
            else:
                x = self.first_conv(x)
                for i in range(self.curLayer):
                    x = self.blocks[i](x)
            x = self.cur_block(x)   # generator
        elif ImagenetSuperProgressiveNASNets.MODE == 'Determined_train':
            x = self.first_conv(x)
            for i in range(self.curLayer):
                x = self.blocks[i](x)
            # assert self.cur_block_mixlayer.n_choices == 1
            x = self.cur_block(x)
            if self.cur_block_mixlayer.enable_mix:
                x = self.cur_block_mixlayer.candidate_ops_mixed_layer(x)
            # x = F.adaptive_avg_pool2d(x, 1)
            # x = x.view(x.size(0), -1)
            batch_size, num_channels, H, W = x.size()
            x = x.view(batch_size, num_channels, -1).mean(dim=2)
            x = self.cur_block_mixlayer.candidate_ops_classifier(x)
        else:   # MODE == 'Finish'
            x = super(ImagenetSuperProgressiveNASNets, self).forward(x)
        return x

    def add_acc(self, acc):
        self.cur_block_mixlayer.accuracy.append(acc)

    def add_infer_latency(self, latency):
        self.cur_block_mixlayer.infer_latency.append(latency)

    @property
    def cur_block(self):
        return self.blocks[self.curLayer]

    @property
    def cur_block_mixlayer(self):
        return self.blocks[self.curLayer].body

    @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    def convert_to_normal_net(self):
        self.init_mix = None
        self.init_classifier = None
        for i, module in enumerate(self.blocks):
            mic = module.mobile_inverted_conv
            if isinstance(mic, MixedEdge):
                assert mic.n_choices == 1
                if i == self.totalSearchLayer - 1:
                    if mic.candidate_ops_mixed_layer is not None:
                        self.feature_mix_layer = mic.candidate_ops_mixed_layer
                    if mic.candidate_ops_classifier is not None \
                            and mic.candidate_ops_classifier.in_features == self.classifier.in_features:
                        self.classifier = mic.candidate_ops_classifier
                module.mobile_inverted_conv = mic.candidate_ops[0]
        assert self.curLayer == self.totalSearchLayer


class Cifar10SuperProgressiveNASNets(Cifar10ProgressiveNASNets):
    MODE = 'Pretrain'  # Pretrain  Search Determined_train Finish

    def __init__(self, depth, alpha, conv_candidates, n_classes=10, bn_param=(0.1, 1e-3), dropout_rate=0,
                 enable_mix=False, share_mix_layer=True, share_classifier=True, enable_init_mix=False,
                 fix_determined=False):

        init_channel = 16
        n = int((depth - 2) / 9)    # 92
        addrate = alpha / (3 * n * 1.0)     # 32
        first_conv = ConvLayer(
            3, init_channel, kernel_size=3, stride=1, use_bn=True, act_func='relu', ops_order='weight_bn_act'
        )

        conv_candidates += ['Zero']
        blocks = []
        in_channel, real_in_channel = init_channel, init_channel
        for i in range(3):
            for j in range(0, n):
                if j == 0 and i != 0:
                    shortcut = PoolingLayer(real_in_channel, real_in_channel, 'avg', kernel_size=2, stride=2)
                    stride = 2
                else:
                    shortcut = IdentityLayer(real_in_channel, real_in_channel)
                    stride = 1
                out_channel = in_channel + addrate
                conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                    conv_candidates, real_in_channel, int(round(out_channel)), stride,
                ), n_classes=n_classes, enable_mix=enable_mix, share_mix_layer=share_mix_layer,
                    share_classifier=share_classifier, dropout_rate=dropout_rate
                )
                block = BottleneckBlock(conv_op, shortcut)
                blocks.append(block)
                real_in_channel = blocks[-1].body.candidate_ops[0].real_outchannels
                in_channel = out_channel

        last_channel = blocks[-1].body.candidate_ops[0].real_outchannels
        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)
        super(Cifar10SuperProgressiveNASNets, self).__init__(first_conv, blocks, classifier,
                                                             enable_init_mix, fix_determined)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    def forward(self, x):
        if Cifar10SuperProgressiveNASNets.MODE == 'Pretrain':
            x = self.first_conv(x)
            for i in range(self.curLayer):
                x = self.blocks[i](x)
            if self.init_mix is not None:
                x = self.init_mix(x)
            # x = F.adaptive_avg_pool2d(x, 1)
            # x = x.view(x.size(0), -1)
            batch_size, num_channels, H, W = x.size()
            x = x.view(batch_size, num_channels, -1).mean(dim=2)
            x = self.init_classifier(x)
        elif Cifar10SuperProgressiveNASNets.MODE == 'Search':
            if self.fix_determined:
                with torch.no_grad():
                    x = self.first_conv(x)
                    for i in range(self.curLayer):
                        x = self.blocks[i](x)
            else:
                x = self.first_conv(x)
                for i in range(self.curLayer):
                    x = self.blocks[i](x)
            x = self.cur_block(x)   # generator
        elif Cifar10SuperProgressiveNASNets.MODE == 'Determined_train':
            x = self.first_conv(x)
            for i in range(self.curLayer):
                x = self.blocks[i](x)
            # assert self.cur_block_mixlayer.n_choices == 1
            x = self.cur_block(x)
            if self.cur_block_mixlayer.enable_mix:
                x = self.cur_block_mixlayer.candidate_ops_mixed_layer(x)
            # x = F.adaptive_avg_pool2d(x, 1)
            # x = x.view(x.size(0), -1)
            batch_size, num_channels, H, W = x.size()
            x = x.view(batch_size, num_channels, -1).mean(dim=2)
            x = self.cur_block_mixlayer.candidate_ops_classifier(x)
        else:   # MODE == 'Finish'
            x = super(Cifar10SuperProgressiveNASNets, self).forward(x)
        return x

    def add_acc(self, acc):
        self.cur_block_mixlayer.accuracy.append(acc)

    def add_infer_latency(self, latency):
        self.cur_block_mixlayer.infer_latency.append(latency)

    @property
    def cur_block(self):
        return self.blocks[self.curLayer]

    @property
    def cur_block_mixlayer(self):
        return self.blocks[self.curLayer].body

    @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    def convert_to_normal_net(self):
        self.init_mix = None
        self.init_classifier = None
        for i, module in enumerate(self.blocks):
            mic = module.bottleneck
            if isinstance(mic, MixedEdge):
                assert mic.n_choices == 1
                if i == self.totalSearchLayer - 1:
                    if mic.candidate_ops_classifier is not None \
                            and mic.candidate_ops_classifier.in_features == self.classifier.in_features:
                        self.classifier = mic.candidate_ops_classifier
                module.bottleneck = mic.candidate_ops[0]
                # module.body = mic.candidate_ops[0]    # the body setter seems not work
        assert self.curLayer == self.totalSearchLayer
