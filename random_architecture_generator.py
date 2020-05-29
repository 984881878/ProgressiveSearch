import os
import tensorwatch as tw
import torch
from models.super_nets.super_progressive import SuperProgressiveNASNets
from modules.mix_op import MixedEdge
import json

arch1 = [17, 16, 10, 10, 16, 11,  4,  8, 14, 17,  9, 12,  3, 12, 12,  6,  8,  9, 12,  1,  1]
arch2 = [17, 17, 16, 10,  8, 11,  7,  8, 16,  1,  1,  7, 16,  5,  2, 13, 17,  5,  2,  5,  1]

arch = arch2

width_stages = [24, 40, 80, 96, 192, 320]
n_cell_stages = [4, 4, 4, 4, 4, 1]
conv_candidates = [
        '3x3_MBConv1', '3x3_MBConv2', '3x3_MBConv3', '3x3_MBConv4', '3x3_MBConv5', '3x3_MBConv6',
        '5x5_MBConv1', '5x5_MBConv2', '5x5_MBConv3', '5x5_MBConv4', '5x5_MBConv5', '5x5_MBConv6',
        '7x7_MBConv1', '7x7_MBConv2', '7x7_MBConv3', '7x7_MBConv4', '7x7_MBConv5', '7x7_MBConv6',
    ]
stride_stages = [2, 2, 2, 1, 2, 1]


def main():
    net = SuperProgressiveNASNets(width_stages, n_cell_stages, conv_candidates, stride_stages,
                                  n_classes=10, width_mult=1, bn_param=(0.1, 1e-3), dropout_rate=0,
                                  enable_mix=False, share_mix_layer=False, share_classifier=False,
                                  enable_init_mix=False)
    for i in arch:
        net.blocks[net.curLayer].mobile_inverted_conv = net.cur_block_mixlayer.candidate_ops[i]
        net.blocks[net.curLayer].is_mixed_edge = isinstance(net.cur_block_mixlayer, MixedEdge)
        net.curLayer = net.curLayer + 1
    assert net.curLayer == net.totalSearchLayer
    SuperProgressiveNASNets.MODE = 'Finish'
    # net.eval()
    # input = torch.rand([1, 3, 32, 32])
    # output = net(input)
    # print(output.shape)
    # drawing = tw.draw_model(net, [1, 3, 32, 32])  # orientation='LR'
    # drawing.save('abc.png')
    json.dump(super(SuperProgressiveNASNets, net).config,
              open('net_config.txt', 'w'), indent=4)


if __name__ == '__main__':
    main()
