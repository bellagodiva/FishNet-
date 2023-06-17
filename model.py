import torch
from torch import nn


#reference: Identity Mappings in Deep Residual Networks (https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua)
#Residual Unit blocks for tail and body&head refining blocks [function M in page 6 equation (6) and (9)]
class Bottleneck(nn.Module):
     def __init__(self, in_channel, out_channel, stride=1, dilation =1, k=1, part=None):
        super(Bottleneck, self).__init__()
        self.part = part
        self.k = k
        self.dilation = dilation
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU()
        bottleneck_channel = out_channel//4
        self.conv1 = nn.Conv2d(in_channel, bottleneck_channel, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(bottleneck_channel)
        # Page 6 Since the up-sampling operation will dilute input features with lower resolution, 
        # we apply dilated convolution in the refining blocks.
        self.conv2 = nn.Conv2d(bottleneck_channel, bottleneck_channel, kernel_size=3, stride=stride, padding=dilation, dilation=dilation)
        self.bn3 = nn.BatchNorm2d(bottleneck_channel)
        self.conv3 = nn.Conv2d(bottleneck_channel, out_channel, kernel_size=1, stride=1, padding=0)

        # Page 8  we still put a 1 × 1 convolution on the skip connection of the last residual blocks for each stage
        # at the tail to change the number of channels between two stages, but we do not use such convolution at the head.
        if self.part == 'tail' and in_channel!=out_channel:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
            )
        else:
            self.shortcut = None
        
     def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        # Page 6 UR-block channel-wise reduction function r in (6)
        # element-wise summation of feature maps from the adjacent k channels to 1 channel.
        if self.part == 'body':
            n, c, h, w = residual.shape
            residual = residual.view(n, c // self.k, self.k, h, w).sum(2)
        # Tail
        elif self.shortcut is not None:
            residual = self.shortcut(residual)

        out += residual
        return out
   

class FishNet(nn.Module):
    def __init__(self, num_cls=10, n_stage=4, n_channel=None, n_res_block=None, n_trans_block=None):
        super(FishNet, self).__init__()
        self.n_channel = n_channel
        self.n_res_block = n_res_block
        self.n_stage = n_stage
        self.n_trans_block = n_trans_block
        self.num_cls = num_cls
        # resolution: 32x32
        in_channel = self.n_channel[0]
        self.conv1 = self.conv_block(3, in_channel // 2)
        self.conv2 = self.conv_block(in_channel // 2, in_channel // 2)
        self.conv3 = self.conv_block(in_channel // 2, in_channel)
        self.pool1 = nn.MaxPool2d(3, padding=1, stride=2)
        # resolution 16x16 with pooll, 32x32 without pool
        self.tail_resblock, self.tail_score, self.tail_resblock_end, self.tail_se  = self.fish_tail()
        self.body_resblock, self.body_transblock, self.body_upsample = self.fish_body()
        self.head_resblock, self.head_transblock, self.head_downsample, self.head_score = self.fish_head()

    def conv_block(self, in_ch, out_ch):
        #resnet 3x3 conv blocks
        return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=False),
                             nn.BatchNorm2d(out_ch),
                             nn.ReLU(inplace=True))

    #reference: https://github.com/kevin-ssy/FishNet/blob/master/models/fishnet.py 
    def score(self, in_ch, out_ch=10, pool=False):
        bn = nn.BatchNorm2d(in_ch)
        relu = nn.ReLU(inplace=True)
        conv_trans = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False)
        bn_out = nn.BatchNorm2d(in_ch // 2)
        conv = nn.Sequential(bn, relu, conv_trans, bn_out, relu)
        if pool:
            fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch // 2, out_ch, kernel_size=1, bias=True))
        else:
            fc = nn.Conv2d(in_ch // 2, out_ch, kernel_size=1, bias=True)
        return [conv, fc]

    #reference: https://github.com/kevin-ssy/FishNet/blob/master/models/fishnet.py 
    def se_block(self, in_ch, out_ch):
        bn = nn.BatchNorm2d(in_ch)
        sq_conv = nn.Conv2d(in_ch, out_ch // 16, kernel_size=1)
        ex_conv = nn.Conv2d(out_ch // 16, out_ch, kernel_size=1)
        return nn.Sequential(bn,
                             nn.ReLU(inplace=True),
                             nn.AdaptiveAvgPool2d(1),
                             sq_conv,
                             nn.ReLU(inplace=True),
                             ex_conv,
                             nn.Sigmoid())

    def fish_tail(self):
        tail_resblock = []
        n_stage = self.n_stage
        n_channel = self.n_channel
        n_res_block = self.n_res_block
        # definition of layer0: 56->28, layer1: 28->14, layer2: 14->7, layer3: 7->7, layer4: 7->14, ... (dimension from the paper:imagenet)
        # n_channel[2]= input channel of layer 2, n_channel[3]= output channel of layer 2
        # layer 0,1,2 (for n_stage=4  in the paper)
        for i in range(0, n_stage-1):
            layers = []
            layers.append(Bottleneck(n_channel[i], n_channel[i+1], part='tail'))
            for j in range(1, n_res_block[i]):
                layers.append(Bottleneck(n_channel[i+1], n_channel[i+1], part='tail'))
            #downsample
            layers.append(nn.MaxPool2d(2, stride=2))
            tail_resblock.append(nn.Sequential(*layers))
        #layer 3 SE_block 
        tail_score = []
        tail_score.extend(self.score(n_channel[n_stage-1], 2*n_channel[n_stage-1], pool=False))
        tail_resblock_e = []
        tail_resblock_e.append(Bottleneck(2*n_channel[n_stage-1], n_channel[n_stage], part='tail'))
        for i in range(1, n_res_block[n_stage-1]):
            tail_resblock_e.append(Bottleneck(n_channel[n_stage], n_channel[n_stage], part='tail'))
        tail_score = nn.Sequential(*tail_score)
        tail_resblock_end = nn.Sequential(*tail_resblock_e)
        tail_se = self.se_block(2*n_channel[n_stage], n_channel[n_stage])
        return nn.ModuleList(tail_resblock), tail_score, tail_resblock_end, tail_se 

    def fish_body(self):
        n_stage = self.n_stage
        n_channel = self.n_channel
        n_res_block = self.n_res_block
        n_trans_block = self.n_trans_block
        self.concat_channel = n_channel.copy()
        # definition of layer0: 56->28, layer1: 28->14, layer2: 14->7, layer3: 7->7, layer4: 7->14, ... (dimension from the paper:imagenet)
        # n_channel[4]= input channel of layer 4, n_channel[5]= output channel of layer 4
        # concat_channel[5] = output channel of layer 4 after concatenation with transfer from input of layer 2
        # 4--2,5--1,6--0 -> layer -- (n_stage-2)-(layer-n_stage)
        for i in range (n_stage+1, 2*n_stage):
            #out = basic_out_chanel + transfer_channel 
            self.concat_channel[i] = n_channel[i] + n_channel[(n_stage-2)-(i-1-n_stage)]
        # layer 4,5,6 (for n_stage=4 in the paper)
        body_resblock, body_transblock = [], []
        for i in range(n_stage, (2*n_stage)-1):
            resblock_layers = []
            k = self.concat_channel[i]//n_channel[i+1]
            dilation = 2 ** (i-n_stage)
            resblock_layers.append(Bottleneck(self.concat_channel[i], n_channel[i+1], dilation=dilation, k=k, part='body'))
            for j in range(1, n_res_block[i]):
                resblock_layers.append(Bottleneck(n_channel[i+1], n_channel[i+1], dilation=dilation))
            body_resblock.append(nn.Sequential(*resblock_layers))
            transblock_layers = []
            #4--2,5--1,6--0 -> transfer layer pair: layer -- (n_stage-2)-(layer-n_stage)
            transfer_layer = (n_stage-2)-(i-n_stage)
            transblock_layers.append(Bottleneck(n_channel[transfer_layer], n_channel[transfer_layer]))
            for j in range(1, n_trans_block[i-n_stage]):
                transblock_layers.append(Bottleneck(n_channel[transfer_layer], n_channel[transfer_layer], dilation=dilation))
            body_transblock.append(nn.Sequential(*transblock_layers))

        body_upsample = nn.Upsample(scale_factor=2)
        return nn.ModuleList(body_resblock), nn.ModuleList(body_transblock), body_upsample

    def fish_head(self):
        n_stage = self.n_stage
        n_channel = self.n_channel
        n_res_block = self.n_res_block
        n_trans_block = self.n_trans_block
        #n_channel[8]= basic output channel of layer 7 before concatenation
        #n_channel[6]= output channel of layer 5
        #concat_channel[8] = output channel of layer 7 after concatenation with transfer from output of layer 5 
        #7--6,8--5,9--4-> transfer layer pair: layer -- ((2*n_stage-1)-1)-(layer-(2*n_stage-1))
        for i in range ((2*n_stage)-1, (3*n_stage)-2):
            #out = basic_out_chanel + transfer_channel
            self.concat_channel[i+1] = n_channel[i+1] + self.concat_channel[((2*n_stage-1)-1)-(i-(2*n_stage-1))]
        #layer 7,8,9 (for n_stage=4 like in the paper)
        head_resblock, head_transblock = [], []
        for i in range((2*n_stage)-1, (3*n_stage)-2):
            resblock_layers = []
            resblock_layers.append(Bottleneck(self.concat_channel[i], n_channel[i+1]))
            for j in range(1, n_res_block[i]):
                resblock_layers.append(Bottleneck(n_channel[i+1], n_channel[i+1]))
            head_resblock.append(nn.Sequential(*resblock_layers))
            transblock_layers = []
            #7--6,8--5,9--4-> transfer layer pair: layer -- ((2*n_stage-1)-1)-(layer-(2*n_stage-1))
            transfer_layer = ((2*n_stage-1)-1)-(i-(2*n_stage-1))
            transblock_layers.append(Bottleneck(self.concat_channel[transfer_layer], self.concat_channel[transfer_layer]))
            for j in range(1, n_trans_block[i-n_stage]):
                transblock_layers.append(Bottleneck(self.concat_channel[transfer_layer], self.concat_channel[transfer_layer]))
            head_transblock.append(nn.Sequential(*transblock_layers))

        head_downsample = nn.MaxPool2d(2, stride=2)
        head_score = []
        head_score.extend(self.score(self.concat_channel[-1], out_ch=self.num_cls, pool=True))
        head_score = nn.Sequential(*head_score)
        return nn.ModuleList(head_resblock), nn.ModuleList(head_transblock), head_downsample, head_score


    def forward(self, x):
        n_stage = self.n_stage
        # save out_feature of every layer to be transferred to body and head
        layer_feature = [None]*len(self.n_channel)

        # input image  CIFAR-10 x resolution n,c,32,32
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #x = self.pool1(x) for resolution 16x16 as input

        layer_feature[0] = x # resolution n,c,32,32 without pool1
        # TAIL (0 to n_stage-1) e.g: n_stage=4 -> 0,1,2
        for i in range(0, n_stage-1):
            out_tail = self.tail_resblock[i](layer_feature[i])
            layer_feature[i+1] = out_tail
        # layer n_stage-1 (e.g: layer 3, resolution 7->1->7)
        score_feat = self.tail_score(out_tail)
        se_feat = self.tail_se(score_feat) 
        out_tail = self.tail_resblock_end(score_feat) * se_feat + se_feat
        layer_feature[n_stage] = out_tail

        # BODY (n_stage to n_stage+(n_stage-1)-1)) e.g: n_stage= 4 -> 4,5,6
        for i in range(n_stage, (2*n_stage)-1):
            #4--2,5--1,6--0 -> transfer layer pair: layer -- (n_stage-2)-(layer-n_stage)
            transfer_layer = (n_stage-2)-(i-n_stage)
            idx = i - n_stage #index start from 0 instead of 4
            basic_feat = self.body_upsample(self.body_resblock[idx](layer_feature[i]))
            transfer_feat = self.body_transblock[idx](layer_feature[transfer_layer])
            concat_feat = torch.cat([basic_feat, transfer_feat], dim=1) #concat channel wise
            layer_feature[i+1] = concat_feat
        
        # HEAD (n_stage+(n_stage-1) to n_stage+(n_stage-1)+(n_stage-1)-1) e.g: n_stage= 4 -> 7,8,9
        for i in range((2*n_stage)-1, (3*n_stage)-2):
            #7--5,8--4,9--3 -> transfer layer pair: layer -- ((2*n_stage-1)-2)-(layer-(2*n_stage-1))
            #however out feature of ith layer is saved in layer_feature[i+1], 7--6,8--5,9--4 corresponding to layer_feature index
            transfer_layer = ((2*n_stage-1)-1)-(i-(2*n_stage-1))
            idx = i - ((2*n_stage)-1) #index start from 0 instead of 7
            basic_feat = self.head_downsample(self.head_resblock[idx](layer_feature[i]))
            transfer_feat = self.head_transblock[idx](layer_feature[transfer_layer])
            concat_feat = torch.cat([basic_feat, transfer_feat], dim=1) #concat channel wise
            layer_feature[i+1] = concat_feat
        score = self.head_score(layer_feature[-1]) 

        # from (n,n_cls,1,1) into (n,n_cls)
        out = score.view(x.size(0), -1)
        return out

