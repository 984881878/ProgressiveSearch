from models.normal_nets.progressive_nets import ImagenetProgressiveNASNets, Cifar10ProgressiveNASNets
from run_manager import RunConfig


def get_net_by_name(name):
    if name == Cifar10ProgressiveNASNets.__name__:
        return Cifar10ProgressiveNASNets
    elif name == ImagenetProgressiveNASNets.__name__:
        return ImagenetProgressiveNASNets
    else:
        raise ValueError('unrecognized type of network: %s' % name)


class ImagenetRunConfig(RunConfig):

    def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='imagenet', train_batch_size=256, test_batch_size=500, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys='bn',
                 model_init='he_fout', init_div_groups=False, validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color='normal', **kwargs):
        super(ImagenetRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            model_init, init_div_groups, validation_frequency, print_frequency
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color

        print(kwargs.keys())

    @property
    def data_config(self):
        return {
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_size': self.valid_size,
            'n_worker': self.n_worker,
            'resize_scale': self.resize_scale,
            'distort_color': self.distort_color,
        }


class Cifar10RunConfig(RunConfig):

    def __init__(self, determined_train=False, determined_train_epoch=10, determined_train_batch=128, n_epochs=20,
                 init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None, dataset='cifar10',
                 train_batch_size=64, test_batch_size=64, valid_size=None, opt_type='sgd', opt_param=None,
                 weight_decay=4e-5, label_smoothing=0.1, no_decay_keys=None, low_lr_parts=None, model_init='he_fout',
                 init_div_groups=False, validation_frequency=1, print_frequency=10, n_worker=16, resize_scale=0.08,
                 distort_color='normal', resize_to_224=False, cutout=False, cutout_length=16, **kwargs):
        super(Cifar10RunConfig, self).__init__(
            determined_train, determined_train_epoch, determined_train_batch,
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing,
            no_decay_keys, low_lr_parts, model_init, init_div_groups,
            validation_frequency, print_frequency
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.resize_to_224 = resize_to_224
        self.cutout = cutout
        self.cutout_length = cutout_length

        print(kwargs.keys())

    @property
    def data_config(self):
        return {
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_size': self.valid_size,
            'n_worker': self.n_worker,
            'resize_scale': self.resize_scale,
            'distort_color': self.distort_color,
            'resize_to_224': self.resize_to_224,
            'cutout': self.cutout,
            'cutout_length': self.cutout_length
        }
