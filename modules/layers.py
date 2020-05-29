from utils import *
from collections import OrderedDict


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        DepthConvLayer.__name__: DepthConvLayer,
        PoolingLayer.__name__: PoolingLayer,
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
        SEMBConv.__name__: SEMBConv,
        BasicBlock.__name__: BasicBlock,
        Bottleneck.__name__: Bottleneck,
        ZeroLayer.__name__: ZeroLayer,
    }

    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class My2DLayer(MyModule):

    def __init__(self, in_channels, out_channels,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        super(My2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm2d(in_channels)
            else:
                modules['bn'] = nn.BatchNorm2d(out_channels)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # weight
        modules['weight'] = self.weight_op()

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def weight_op(self):
        raise NotImplementedError

    """ Methods defined in MyModule """

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError

    @staticmethod
    def is_zero_layer():
        return False


class ConvLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        super(ConvLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict['conv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
            dilation=self.dilation, groups=self.groups, bias=self.bias
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict['shuffle'] = ShuffleLayer(self.groups)

        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                return '%dx%d_DilatedConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                return '%dx%d_DilatedGroupConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_GroupConv' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': ConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            **super(ConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)

    def get_flops(self, x):
        return count_conv_flop(self.conv, x), self.forward(x)


class DepthConvLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        super(DepthConvLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict['depth_conv'] = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
            dilation=self.dilation, groups=self.in_channels, bias=False
        )
        weight_dict['point_conv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1, groups=self.groups, bias=self.bias
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict['shuffle'] = ShuffleLayer(self.groups)
        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.dilation > 1:
            return '%dx%d_DilatedDepthConv' % (kernel_size[0], kernel_size[1])
        else:
            return '%dx%d_DepthConv' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': DepthConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            **super(DepthConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return DepthConvLayer(**config)

    def get_flops(self, x):
        depth_flop = count_conv_flop(self.depth_conv, x)
        x = self.depth_conv(x)
        point_flop = count_conv_flop(self.point_conv, x)
        x = self.point_conv(x)
        return depth_flop + point_flop, self.forward(x)


class PoolingLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 pool_type, kernel_size=2, stride=2,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride

        super(PoolingLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        if self.stride == 1:
            # same padding if `stride == 1`
            padding = get_same_padding(self.kernel_size)
        else:
            padding = 0

        weight_dict = OrderedDict()
        if self.pool_type == 'avg':
            weight_dict['pool'] = nn.AvgPool2d(
                self.kernel_size, stride=self.stride, padding=padding, count_include_pad=False
            )
        elif self.pool_type == 'max':
            weight_dict['pool'] = nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=padding)
        else:
            raise NotImplementedError
        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        return '%dx%d_%sPool' % (kernel_size[0], kernel_size[1], self.pool_type.upper())

    @property
    def config(self):
        return {
            'name': PoolingLayer.__name__,
            'pool_type': self.pool_type,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            **super(PoolingLayer, self).config
        }

    @staticmethod
    def build_from_config(config):
        return PoolingLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)


class IdentityLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(IdentityLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        return None

    @property
    def module_str(self):
        return 'Identity'

    @property
    def config(self):
        return {
            'name': IdentityLayer.__name__,
            **super(IdentityLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)


class LinearLayer(MyModule):

    def __init__(self, in_features, out_features, bias=True,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm1d(in_features)
            else:
                modules['bn'] = nn.BatchNorm1d(out_features)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # linear
        modules['weight'] = {'linear': nn.Linear(self.in_features, self.out_features, self.bias)}

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        return '%dx%d_Linear' % (self.in_features, self.out_features)

    @property
    def config(self):
        return {
            'name': LinearLayer.__name__,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'bias': self.bias,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)

    def get_flops(self, x):
        return self.linear.weight.numel(), self.forward(x)

    @staticmethod
    def is_zero_layer():
        return False


class MBInvertedConvLayer(MyModule):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=6, mid_channels=None):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('act', nn.ReLU6(inplace=True)),
            ]))

        pad = get_same_padding(self.kernel_size)
        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=feature_dim, bias=False)),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', nn.ReLU6(inplace=True)),
        ]))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def real_outchannels(self):
        return self.out_channels

    @property
    def module_str(self):
        return '%dx%d_MBConv%d' % (self.kernel_size, self.kernel_size, self.expand_ratio)

    @property
    def config(self):
        return {
            'name': MBInvertedConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
        }

    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)

    @staticmethod
    def is_zero_layer():
        return False

    def get_flops(self, x):
        if self.inverted_bottleneck:
            flop1 = count_conv_flop(self.inverted_bottleneck.conv, x)
            x = self.inverted_bottleneck(x)
        else:
            flop1 = 0

        flop2 = count_conv_flop(self.depth_conv.conv, x)
        x = self.depth_conv(x)

        flop3 = count_conv_flop(self.point_linear.conv, x)
        x = self.point_linear(x)

        return flop1 + flop2 + flop3, x


class SEMBConv(nn.Module):
    outchannel_ratio = 4

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=6, mid_channels=None,
                 se_reduction_ratio=2):
        super(SEMBConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.se_reduction_ratio = se_reduction_ratio

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('act', nn.ReLU6(inplace=True)),
            ]))

        pad = get_same_padding(self.kernel_size)
        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=feature_dim, bias=False)),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', nn.ReLU6(inplace=True)),
        ]))

        if se_reduction_ratio != 0:
            self.se_conv = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(feature_dim, feature_dim // se_reduction_ratio, bias=True)),
                ('act1', nn.ReLU6(inplace=True)),
                ('fc2', nn.Linear(feature_dim // se_reduction_ratio, feature_dim, bias=True)),
                ('act2', nn.Sigmoid())
            ]))
        else:
            self.se_conv = None

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, out_channels * SEMBConv.outchannel_ratio, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels * SEMBConv.outchannel_ratio)),
        ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        if self.se_conv:
            batch_size, num_channels, H, W = x.size()
            y = x.view(batch_size, num_channels, -1).mean(dim=2)
            y = self.se_conv(y)
            x = torch.mul(x, y.view(batch_size, num_channels, 1, 1))
        x = self.point_linear(x)
        return x

    @property
    def real_outchannels(self):
        return self.out_channels * SEMBConv.outchannel_ratio

    @property
    def module_str(self):
        return '%dx%d_SEMBConv%d_%d_%d' % (
            self.kernel_size, self.kernel_size, self.expand_ratio, self.se_reduction_ratio, SEMBConv.outchannel_ratio
        )

    @property
    def config(self):
        return {
            'name': SEMBConv.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
            'se_reduction_ratio': self.se_reduction_ratio,
            'outchannel_ratio': SEMBConv.outchannel_ratio
        }

    @staticmethod
    def build_from_config(config):
        config.pop('outchannel_ratio')
        return SEMBConv(**config)

    @staticmethod
    def is_zero_layer():
        return False

    def get_flops(self, x):
        if self.inverted_bottleneck:
            flop1 = count_conv_flop(self.inverted_bottleneck.conv, x)
            x = self.inverted_bottleneck(x)
        else:
            flop1 = 0

        flop2 = count_conv_flop(self.depth_conv.conv, x)
        x = self.depth_conv(x)

        if self.se_conv:
            flop3 = (self.se_conv.fc1.in_features + 1) * self.se_conv.fc1.out_features
            flop3 = 2 * flop3 + self.se_conv.fc2.out_features - self.se_conv.fc2.in_features
        else:
            flop3 = 0

        flop4 = count_conv_flop(self.point_linear.conv, x)
        x = self.point_linear(x)

        return flop1 + flop2 + flop3 + flop4, x


class BasicBlock(MyModule):
    outchannel_ratio = 4    # or 1

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=1, mid_channels=None):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels

        # expand_ratio in [0, 1]
        if self.mid_channels is None:
            feature_dim = max(self.in_channels, self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 0:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('bn1', nn.BatchNorm2d(in_channels)),
                ('conv', nn.Conv2d(in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn2', nn.BatchNorm2d(feature_dim)),
                ('act', nn.ReLU(inplace=True)),
            ]))

        # kernel_size in [3, 5, 7]
        pad = get_same_padding(kernel_size)
        self.normal_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, bias=False)),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', nn.ReLU(inplace=True)),
        ]))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, out_channels * BasicBlock.outchannel_ratio, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels * BasicBlock.outchannel_ratio)),
        ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.normal_conv(x)
        x = self.point_linear(x)
        return x

    def get_flops(self, x):
        if self.inverted_bottleneck:
            flop1 = count_conv_flop(self.inverted_bottleneck.conv, x)
            x = self.inverted_bottleneck(x)
        else:
            flop1 = 0

        flop2 = count_conv_flop(self.normal_conv.conv, x)
        x = self.normal_conv(x)

        flop3 = count_conv_flop(self.point_linear.conv, x)
        x = self.point_linear(x)

        return flop1 + flop2 + flop3, x

    @property
    def real_outchannels(self):
        return BasicBlock.outchannel_ratio * self.out_channels

    @property
    def module_str(self):
        return 'BasicBlock_%dx%dConv_Expand%d_OUTratio%d' % (
            self.kernel_size, self.kernel_size, self.expand_ratio, BasicBlock.outchannel_ratio
        )

    @property
    def config(self):
        return {
            'name': BasicBlock.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
            'outchannel_ratio': BasicBlock.outchannel_ratio,
        }

    @staticmethod
    def build_from_config(config):
        config.pop('outchannel_ratio')
        return BasicBlock(**config)

    @staticmethod
    def is_zero_layer():
        return False


class Bottleneck(MyModule):
    outchannel_ratio = 4

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=1, mid_channels=None,
                 se_reduction_ratio=2):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.se_reduction_ratio = se_reduction_ratio

        # expand_ratio in [0, 1, 2]
        if self.mid_channels is None:
            feature_dim = max(self.in_channels, self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 0:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('bn1', nn.BatchNorm2d(in_channels)),
                ('conv', nn.Conv2d(in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn2', nn.BatchNorm2d(feature_dim)),
                ('act', nn.ReLU(inplace=True)),
            ]))

        # kernel_size in [3, 5, 7]
        pad = get_same_padding(kernel_size)
        self.normal_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, bias=False)),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', nn.ReLU(inplace=True)),
        ]))

        self.se_conv = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(feature_dim, feature_dim // se_reduction_ratio, bias=True)),
            ('act1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(feature_dim // se_reduction_ratio, feature_dim, bias=True)),
            ('act2', nn.Sigmoid())
        ]))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, out_channels * Bottleneck.outchannel_ratio, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels * Bottleneck.outchannel_ratio)),
        ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.normal_conv(x)

        """
        ChannelSELayer implemented in 
        https://github.com/ai-med/squeeze_and_excitation/blob/master/squeeze_and_excitation/squeeze_and_excitation.py
        """
        batch_size, num_channels, H, W = x.size()
        y = x.view(batch_size, num_channels, -1).mean(dim=2)
        y = self.se_conv(y)
        x = torch.mul(x, y.view(batch_size, num_channels, 1, 1))

        x = self.point_linear(x)
        return x

    def get_flops(self, x):
        if self.inverted_bottleneck:
            flop1 = count_conv_flop(self.inverted_bottleneck.conv, x)
            x = self.inverted_bottleneck(x)
        else:
            flop1 = 0

        flop2 = count_conv_flop(self.normal_conv.conv, x)
        x = self.normal_conv(x)

        flop3 = (self.se_conv.fc1.in_features + 1) * self.se_conv.fc1.out_features
        # flop4 approximately equal to flop3
        # flop4 = (self.se_conv.fc2.in_features + 1) * self.se_conv.fc2.out_features
        flop4 = flop3 + self.se_conv.fc2.out_features - self.se_conv.fc2.in_features

        flop5 = count_conv_flop(self.point_linear.conv, x)
        x = self.point_linear(x)

        return flop1 + flop2 + flop3 + flop4 + flop5, x

    @property
    def real_outchannels(self):
        return Bottleneck.outchannel_ratio * self.out_channels

    @property
    def module_str(self):
        return 'Bottleneck_%dx%dConv_Expand%d_OUTratio%d_SEratio%d' % (
            self.kernel_size, self.kernel_size, self.expand_ratio, Bottleneck.outchannel_ratio, self.se_reduction_ratio
        )

    @property
    def config(self):
        return {
            'name': Bottleneck.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
            'outchannel_ratio': Bottleneck.outchannel_ratio,
            'se_reduction_ratio': self.se_reduction_ratio
        }

    @staticmethod
    def build_from_config(config):
        config.pop('outchannel_ratio')
        return Bottleneck(**config)

    @staticmethod
    def is_zero_layer():
        return False


class ZeroLayer(MyModule):

    def __init__(self, in_channels, out_channels, stride):
        super(ZeroLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        assert c == self.in_channels
        device = x.get_device() if x.is_cuda else torch.device('cpu')
        # noinspection PyUnresolvedReferences
        padding = torch.zeros(n, self.out_channels, h, w, device=device, requires_grad=False)
        return padding

    @property
    def module_str(self):
        return 'Zero'

    @property
    def config(self):
        return {
            'name': ZeroLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'stride': self.stride,
        }

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)

    @staticmethod
    def is_zero_layer():
        return True
