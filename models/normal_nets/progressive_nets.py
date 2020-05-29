from modules.layers import *
import json
from modules.mix_op import MixedEdge


# def proxyless_base(net_config=None, n_classes=1000, bn_param=(0.1, 1e-3), dropout_rate=0):
#     assert net_config is not None, 'Please input a network config'
#     net_config_path = download_url(net_config)
#     net_config_json = json.load(open(net_config_path, 'r'))
#
#     net_config_json['classifier']['out_features'] = n_classes
#     net_config_json['classifier']['dropout_rate'] = dropout_rate
#
#     net = ProgressiveNASNets.build_from_config(net_config_json)
#     net.set_bn_param(momentum=bn_param[0], eps=bn_param[1])
#
#     return net


class MobileInvertedResidualBlock(MyModule):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut
        self.is_mixed_edge = isinstance(self.mobile_inverted_conv, MixedEdge)
        if self.is_mixed_edge:
            self.mobile_inverted_conv.residual = True if shortcut is not None else False
            self.mobile_inverted_conv.mode = 'imagenet'
            # self.mobile_inverted_conv.shortcut = 'identity' if shortcut is not None else None

    def forward(self, x):
        res = self.mobile_inverted_conv(x)
        if self.is_mixed_edge and self.mobile_inverted_conv.n_choices > 1:
            return res
        if self.is_mixed_edge:
            res, = res
        if self.shortcut is not None:
            skip_x = self.shortcut(x)
            res = res + skip_x
        return res

    @property
    def body(self):
        return self.mobile_inverted_conv

    # @body.setter
    # def body(self, val):
    #     self.mobile_inverted_conv = val

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str, self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

    def get_flops(self, x):
        flops1, conv_x = self.mobile_inverted_conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)


class BottleneckBlock(MyModule):

    def __init__(self, bottleneck, shortcut):
        super(BottleneckBlock, self).__init__()

        self.bottleneck = bottleneck
        self.shortcut = shortcut
        self.is_mixed_edge = isinstance(self.bottleneck, MixedEdge)
        if self.is_mixed_edge:
            self.bottleneck.residual = True if shortcut is not None else False
            self.bottleneck.mode = 'cifar10'
            shortcut_type = 'identity' if isinstance(shortcut, IdentityLayer) else 'avgpool'
            self.bottleneck.shortcut = shortcut_type

    def forward(self, x):
        res = self.bottleneck(x)
        if self.is_mixed_edge and self.bottleneck.n_choices > 1:
            return res
        if self.is_mixed_edge:
            res, = res
        if self.shortcut is not None:
            skip_x = self.shortcut(x)
        else:
            skip_x = x

        batch_size, residual_channel, featuremap_size = res.size()[0], res.size()[1], res.size()[2:4]
        shortcut_channel = skip_x.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0))
            res += torch.cat((skip_x, padding), 1)
        else:
            res += skip_x

        return res

    @property
    def body(self):
        return self.bottleneck

    # @body.setter
    # def body(self, val):
    #     self.bottleneck = val

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.bottleneck.module_str, self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': BottleneckBlock.__name__,
            'bottleneck': self.bottleneck.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None
        }

    @staticmethod
    def build_from_config(config):
        bottleneck = set_layer_from_config(config['bottleneck'])
        shortcut = set_layer_from_config(config['shortcut'])
        return BottleneckBlock(bottleneck, shortcut)

    def get_flops(self, x):
        flops1, conv_x = self.bottleneck.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)


class ImagenetProgressiveNASNets(MyNetwork):

    def __init__(self, first_conv, blocks, feature_mix_layer, classifier, enable_init_mix, fix_determined):
        super(ImagenetProgressiveNASNets, self).__init__()

        self.first_conv = first_conv

        channels = blocks[0].mobile_inverted_conv.out_channels
        self.init_mix = None
        if enable_init_mix:
            self.init_mix = ConvLayer(
                channels, channels * 4,
                kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act',
            )
            channels = channels * 4
        self.init_classifier = LinearLayer(channels, classifier.out_features, dropout_rate=classifier.dropout_rate)

        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier

        self.curLayer = 0
        self.totalSearchLayer = len(blocks)  # the first_block in blocks is count

        self.init_cur_layer()
        self.fix_determined = fix_determined

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        # x = self.global_avg_pooling(x)
        # x = x.view(x.size(0), -1)  # flatten
        batch_size, num_channels, H, W = x.size()
        x = x.view(batch_size, num_channels, -1).mean(dim=2)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ''
        _str += self.first_conv.module_str
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.feature_mix_layer.module_str
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': ImagenetProgressiveNASNets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'feature_mix_layer': self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])
        blocks = []
        for block_config in config['blocks']:
            assert block_config['name'] == 'MobileInvertedResidualBlock'
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = ImagenetProgressiveNASNets(first_conv, blocks, feature_mix_layer, classifier, enable_init_mix=False,
                                         fix_determined=False)
        net.init_classifier = None
        assert net.curLayer == net.totalSearchLayer
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def get_flops(self, x):
        flop, x = self.first_conv.get_flops(x)

        for i in range(self.curLayer):
            delta_flop, x = self.blocks[i].get_flops(x)
            flop += delta_flop

        if self.curLayer != self.totalSearchLayer:
            return flop, x

        delta_flop, x = self.feature_mix_layer.get_flops(x)
        flop += delta_flop

        # x = self.global_avg_pooling(x)
        # x = x.view(x.size(0), -1)  # flatten
        batch_size, num_channels, H, W = x.size()
        x = x.view(batch_size, num_channels, -1).mean(dim=2)

        delta_flop, x = self.classifier.get_flops(x)
        flop += delta_flop
        return flop, x

    def init_cur_layer(self):
        self.curLayer = 0
        for i in range(self.totalSearchLayer):
            if self.blocks[i].is_mixed_edge and self.blocks[i].body.n_choices > 1:
                break
            self.curLayer = self.curLayer + 1


class Cifar10ProgressiveNASNets(MyNetwork):

    def __init__(self, first_conv, blocks, classifier, enable_init_mix, fix_determined):
        super(Cifar10ProgressiveNASNets, self).__init__()

        self.first_conv = first_conv

        channels = first_conv.out_channels
        self.init_mix = None
        if enable_init_mix:
            self.init_mix = ConvLayer(
                channels, channels * 4,
                kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act',
            )
            channels = channels * 4
        self.init_classifier = LinearLayer(channels, classifier.out_features, dropout_rate=classifier.dropout_rate)

        self.blocks = nn.ModuleList(blocks)

        self.bn_final = nn.BatchNorm2d(classifier.in_features)
        self.relu_final = nn.ReLU(inplace=True)
        self.classifier = classifier

        self.curLayer = 0
        self.totalSearchLayer = len(blocks)  # the first_block in blocks is count

        self.init_cur_layer()
        self.fix_determined = fix_determined

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        # x = self.feature_mix_layer(x)
        # x = self.global_avg_pooling(x)
        # x = x.view(x.size(0), -1)  # flatten
        x = self.bn_final(x)
        x = self.relu_final(x)
        batch_size, num_channels, H, W = x.size()
        x = x.view(batch_size, num_channels, -1).mean(dim=2)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ''
        _str += self.first_conv.module_str
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': Cifar10ProgressiveNASNets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        classifier = set_layer_from_config(config['classifier'])
        blocks = []
        for block_config in config['blocks']:
            assert block_config['name'] == 'BottleneckBlock'
            blocks.append(BottleneckBlock.build_from_config(block_config))

        net = Cifar10ProgressiveNASNets(first_conv, blocks, classifier, enable_init_mix=False, fix_determined=False)
        net.init_classifier = None
        assert net.curLayer == net.totalSearchLayer
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def get_flops(self, x):
        flop, x = self.first_conv.get_flops(x)

        for i in range(self.curLayer):
            delta_flop, x = self.blocks[i].get_flops(x)
            flop += delta_flop

        if self.curLayer != self.totalSearchLayer:
            return flop, x

        # x = self.global_avg_pooling(x)
        # x = x.view(x.size(0), -1)  # flatten
        batch_size, num_channels, H, W = x.size()
        x = x.view(batch_size, num_channels, -1).mean(dim=2)

        delta_flop, x = self.classifier.get_flops(x)
        flop += delta_flop
        return flop, x

    def init_cur_layer(self):
        self.curLayer = 0
        for i in range(self.totalSearchLayer):
            if self.blocks[i].is_mixed_edge and self.blocks[i].body.n_choices > 1:
                break
            self.curLayer = self.curLayer + 1
