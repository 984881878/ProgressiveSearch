import argparse
import time
# from models import ImagenetRunConfig
# from nas_manager import ProgressiveArchSearchConfig
from models import Cifar10RunConfig
from nas_manager import *
from models.super_nets.super_progressive import ImagenetSuperProgressiveNASNets as SuperProgressiveNASNets

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='Imagenet_EXP/%s' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
parser.add_argument('--gpu', help='gpu available', default='0,1,2,3')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--manual_seed', default=0, type=int)

""" run config """
parser.add_argument('--n_epochs', type=int, default=5, help='how many epoches each layer to be trained')
parser.add_argument('--init_lr', type=float, default=0.025)
parser.add_argument('--lr_schedule_type', type=str, default='cosine')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['imagenet', 'cifar10'])
parser.add_argument('--resize_to_224', action='store_true', help='resize cifar10 img to 224x224')
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=128)
parser.add_argument('--valid_size', type=int, default=None)
parser.add_argument('--resize_scale', type=float, default=0.08)
parser.add_argument('--distort_color', type=str, default='normal', choices=['normal', 'strong', 'None'])
parser.add_argument('--cutout', action='store_true')
parser.add_argument('--cutout_length', type=int, default=16)
parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--no_decay_keys', type=str, default=None, choices=[None, 'bn', 'bn#bias'])
parser.add_argument('--low_lr_parts', type=str, default=None, choices=[None, 'determined'])     # !!!!!!!!!!!!!!!!!!!!!!
parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=10)
parser.add_argument('--n_worker', type=int, default=32)

""" net config """
parser.add_argument('--width_stages', type=str, default='24,40,80,96,192,320')
parser.add_argument('--n_cell_stages', type=str, default='4,4,4,4,4,1')
parser.add_argument('--stride_stages', type=str, default='2,2,2,1,2,1')
parser.add_argument('--width_mult', type=float, default=1.0)
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--bn_eps', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--enable_mix', action='store_true')
parser.add_argument('--share_mix_layer', action='store_true')
parser.add_argument('--share_classifier', action='store_true')
parser.add_argument('--enable_init_mix', action='store_true')
parser.add_argument('--fix_determined', action='store_true')

parser.add_argument('--pretrained_epochs', type=int, default=5)
parser.add_argument('--pretrained_batch', type=int, default=128)
parser.add_argument('--determined_train', action='store_true')
parser.add_argument('--re_init', type=str, default='last_determined', choices=['last_determined', 'determined_part', 'no_reinit'])
parser.add_argument('--determined_train_epoch', type=int, default=5)
parser.add_argument('--determined_train_batch', type=int, default=128)

""" shared hyper-parameters """
parser.add_argument('--target_hardware', type=str, default=None, choices=['cpu', 'gpu1', None])
parser.add_argument('--reg_loss_type', type=str, default='add', choices=['add', 'mul'])
parser.add_argument('--reg_loss_acc', type=float, default=0.9)  # grad_reg_loss_params
parser.add_argument('--reg_loss_latency', type=float, default=0.1)  # grad_reg_loss_params

if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    os.makedirs(args.path, exist_ok=True)

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    run_config = Cifar10RunConfig(
        **args.__dict__
    )

    width_stages_str = '-'.join(args.width_stages.split(','))
    # build net from args
    args.width_stages = [int(val) for val in args.width_stages.split(',')]
    args.n_cell_stages = [int(val) for val in args.n_cell_stages.split(',')]
    args.stride_stages = [int(val) for val in args.stride_stages.split(',')]
    args.conv_candidates = [
        '3x3_MBConv1', '3x3_MBConv2', '3x3_MBConv3', '3x3_MBConv4', '3x3_MBConv5', '3x3_MBConv6',
        '5x5_MBConv1', '5x5_MBConv2', '5x5_MBConv3', '5x5_MBConv4', '5x5_MBConv5', '5x5_MBConv6',
        '7x7_MBConv1', '7x7_MBConv2', '7x7_MBConv3', '7x7_MBConv4', '7x7_MBConv5', '7x7_MBConv6',
    ]
    super_net = SuperProgressiveNASNets(
        width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
        conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
        bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout, enable_mix=args.enable_mix,
        share_mix_layer=args.share_mix_layer, share_classifier=args.share_classifier,
        enable_init_mix=args.enable_init_mix, fix_determined=args.fix_determined
    )

    if args.reg_loss_type == 'add':
        args.reg_loss_params = {
            'w_acc': args.reg_loss_acc,
            'w_latency': args.reg_loss_latency,
        }
    arch_search_config = ProgressiveArchSearchConfig(**args.__dict__)

    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))
    print('Architecture Search config:')
    for k, v in arch_search_config.config.items():
        print('\t%s: %s' % (k, v))

    # arch search run manager
    arch_search_run_manager = ArchSearchRunManager(args.path, super_net, run_config, arch_search_config)

    # resume
    if args.resume:
        try:
            arch_search_run_manager.load_model()
        except Exception as e:
            print('fail to load models: %s' % e)

    if arch_search_run_manager.pretrain:
        arch_search_run_manager.init_train(pretrain_epochs=args.pretrained_epochs,
                                           pretrained_batch=args.pretrained_batch)

    # training
    arch_search_run_manager.train(args.re_init)
