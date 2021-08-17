import math

import torch
import torch.nn as nn

from stylegan2.model import EqualLinear

MAX_LAYERS = 18
DEFUALT_COMMON_LAYERS = 5
DEFAULT_INDEX_BITS = 10


class LatentBlender(nn.Module):
    def __init__(self, size, index_bits=DEFAULT_INDEX_BITS, extras=0):
        super(LatentBlender, self).__init__()
        layers = []

        self.size = size
        self.index_bits = index_bits

        sizes = [size * 2 + index_bits * (MAX_LAYERS + 1) + extras * size, size * 5, size * 5, size * 5, size, size]

        for i in range(len(sizes) - 1):
            layers.append(
                EqualLinear(
                    sizes[i], sizes[i + 1], lr_mul=0.01, activation='fused_lrelu'
                )
            )

        self.disentangle_net = nn.Sequential(*layers)

    def forward(self, a, b, i=None, rgb=False, extra=None):
        x = torch.cat((a[0], b[0]))
        if not extra is None:
            x = torch.cat((x, extra))
        if not (i is None):
            indicator = torch.zeros(MAX_LAYERS + 1, device="cuda:0")
            indicator[i] = 1
            if rgb:
                indicator[-1] = 1
            x = torch.cat([x, indicator.repeat(self.index_bits)])
        x = self.disentangle_net(x)
        x = torch.sigmoid(x)
        x = x.unsqueeze(0)

        return a[0] + x * (b[0] - a[0])  # Fusion procedure


class FusionNet(nn.Module):
    def __init__(self, index_bits=DEFAULT_INDEX_BITS, common_layers=DEFUALT_COMMON_LAYERS):
        super(FusionNet, self).__init__()

        self.max_pools = [nn.MaxPool1d(512 // x) for x in [512, 256, 128, 64, 32]]
        self.upsample = nn.Upsample(size=512, mode='nearest')

        self.blender_segments = LatentBlender(512, index_bits=index_bits, extras=1)
        self.blender_common = LatentBlender(512, index_bits=index_bits)
        self.common_layers = common_layers

    def up(self, x):
        return self.upsample(x.unsqueeze(0)).squeeze(0)

    def pool(self, x, size):
        size_index = 9 - int(math.log2(size))
        return self.max_pools[size_index](x.unsqueeze(0)).squeeze(0)

    def forward(self, s0, s1, s2, only_common=False, only_segments=False):
        if only_common:
            assert s1 is None
        if only_segments:
            assert s2 is None
        res = [[],[]]
        for rgb in [0, 1]:
            for i in range(len(s0[rgb])):
                s0_up = self.up(s0[rgb][i])
                if not only_common:
                    s1_up = self.up(s1[rgb][i])
                if not only_segments:
                    s2_up = self.up(s2[rgb][i])

                skip_common_layer = False
                if not (self.common_layers is None):
                    skip_common_layer = i >= self.common_layers

                if only_common:
                    if skip_common_layer:
                        res[rgb].append(s0[rgb][i])
                    else:
                        x = self.blender_common(s0_up, s2_up, i=i, rgb=(rgb==1))
                        res[rgb].append(self.pool(x, s0[rgb][i].size()[1]))
                elif only_segments or skip_common_layer:
                    x = self.blender_segments(s0_up, s1_up, i=i, rgb=False, extra=torch.zeros_like(s1_up[0]))
                    res[rgb].append(self.pool(x, s0[rgb][i].size()[1]))
                else:
                    x0 = self.blender_common(s0_up, s2_up, i=i, rgb=False)
                    x0 = self.pool(x0, s0[rgb][i].size()[1])
                    x1 = self.blender_common(s1_up, s2_up, i=i, rgb=False)
                    x1 = self.pool(x1, s0[rgb][i].size()[1])

                    x = self.blender_segments(self.up(x0), self.up(x1), i=i, rgb=False, extra=s2_up[0])
                    res[rgb].append(self.pool(x, s0[rgb][i].size()[1]))

        return res
