from aawedha.models.pytorch.torch_utils import LineardWithConstraint
from aawedha.models.pytorch.torch_utils import Conv2dWithConstraint
from aawedha.models.pytorch.torchmodel import TorchModel
from aawedha.layers.cmt import Attention, Mlp
from torch import nn, flatten
import torch.nn.functional as F
import torch


class EEGNat(TorchModel):

    def __init__(self, nb_classes, Chans=64, Samples=128,  
                 F1=4, D=2, kernLength=64, heads=1, dropout=0.5, 
                 device="cuda", name="EEGNat"):

        super().__init__(device=device, name=name)
        regRate = .25
        
        F2 = F1*D
        # EEGNET
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn1   = nn.BatchNorm2d(F1)
        self.conv2 = Conv2dWithConstraint(F1, F1 * D, (Chans, 1), max_norm=1, bias=False, groups=F1, padding="valid")      
        self.bn2   = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(p=dropout)
        # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
        self.conv_sep_depth = nn.Conv2d(F2, F2, (1, 16), bias=False, groups=F2, padding="same")
        self.conv_sep_point = nn.Conv2d(F2, F2, (1, 1), bias=False, padding="valid")
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(p=dropout)
        # attention
        self.ln1   = nn.LayerNorm(F2)
        pos_shape = Samples // 32
        self.relative_pos = nn.Parameter(torch.randn(heads, pos_shape, pos_shape)) 
        self.attn   = Attention(dim=F2, num_heads=heads, qkv_bias=False, 
                                      qk_scale=None, attn_drop=0., proj_drop=0., 
                                      qk_ratio=1, sr_ratio=1)
        #
        self.dense = LineardWithConstraint(F2*pos_shape, nb_classes, max_norm=regRate)
        #
        self.init_weights()

    def forward(self, x):
        x = self._reshape_input(x)     
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))   
        x = F.elu(x)
        x = self.drop1(self.pool1(x))        
        x = self.conv_sep_point(self.conv_sep_depth(x))        
        x = self.bn3(x)
        x = F.elu(x)
        x = self.drop2(self.pool2(x))
        #
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1) # BCHW->BNC (N=H*W)
        x = self.attn(self.ln1(x), H, W, self.relative_pos)     
        x = flatten(x, 1)        
        x = self.dense(x)
        return x


class EEGNat2(TorchModel):

    def __init__(self, nb_classes, Chans=64, Samples=128, 
                  dropout=0.5, F1=4, D=2, kernLength=64, heads=1,
                  device="cuda", name="EEGNat2"):

        super().__init__(device=device, name=name)
        regRate = .25
        
        F2 = F1*D
        # EEGNET
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn1   = nn.BatchNorm2d(F1)
        self.conv2 = Conv2dWithConstraint(F1, F1 * D, (Chans, 1), max_norm=1, bias=False, groups=F1, padding="valid")      
        self.bn2   = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(p=dropout)
        # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
        self.conv_sep_depth = nn.Conv2d(F2, F2, (1, 16), bias=False, groups=F2, padding="same")
        self.conv_sep_point = nn.Conv2d(F2, F2, (1, 1), bias=False, padding="valid")
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(p=dropout)
        # attention
        self.ln1   = nn.LayerNorm(F2)
        pos_shape = Samples // 32
        self.relative_pos = nn.Parameter(torch.randn(heads, pos_shape, pos_shape)) 
        self.attn   = Attention(dim=F2, num_heads=heads, qkv_bias=False, 
                                      qk_scale=None, attn_drop=0., proj_drop=0., 
                                      qk_ratio=1, sr_ratio=1)
        self.ln2   = nn.LayerNorm(F2)
        self.mlp   = Mlp(F2) 
        #
        self.dense = LineardWithConstraint(F2*pos_shape, nb_classes, max_norm=regRate)
        #
        self.init_weights()

    def forward(self, x):
        x = self._reshape_input(x)     
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))   
        x = F.elu(x)
        x = self.drop1(self.pool1(x))        
        x = self.conv_sep_point(self.conv_sep_depth(x))        
        x = self.bn3(x)
        x = F.elu(x)
        x = self.drop2(self.pool2(x))
        #
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1) # BCHW->BNC (N=H*W)
        x = self.attn(self.ln1(x), H, W, self.relative_pos)  
        x = self.mlp(self.ln2(x), H, W)   
        x = flatten(x, 1)        
        x = self.dense(x)
        return x

class EEGNat3(TorchModel):

    def __init__(self, nb_classes, Chans=64, Samples=128,
                 dropout=0.5, F1=4, D=2, kernLength=64, heads=1,
                 device="cuda", name="EEGNat"):

        super().__init__(device=device, name=name)
        regRate = .25
        F2 = F1*D
        # EEGNET
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn1   = nn.BatchNorm2d(F1)
        self.conv2 = Conv2dWithConstraint(F1, F1 * D, (Chans, 1), max_norm=1, bias=False, groups=F1, padding="valid")      
        self.bn2   = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(p=dropout)
        # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
        self.conv_sep_depth = nn.Conv2d(F2, F2, (1, 16), bias=False, groups=F2, padding="same")
        self.conv_sep_point = nn.Conv2d(F2, F2, (1, 1), bias=False, padding="valid")
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(p=dropout)
        # attention
        pos_shape = Samples // 32
        self.ln1   = nn.LayerNorm(F2)        
        self.relative_pos = nn.Parameter(torch.randn(heads, pos_shape, pos_shape)) 
        self.attn   = Attention(dim=F2, num_heads=heads, qkv_bias=False, 
                                      qk_scale=None, attn_drop=0., proj_drop=0., 
                                      qk_ratio=1, sr_ratio=1)
        #
        self.ln2   = nn.LayerNorm(F2)
        self.relative_pos2 = nn.Parameter(torch.randn(heads, pos_shape, pos_shape)) 
        self.attn2   = Attention(dim=F2, num_heads=heads, qkv_bias=False, 
                                      qk_scale=None, attn_drop=0., proj_drop=0., 
                                      qk_ratio=1, sr_ratio=1)        
        #
        self.dense = LineardWithConstraint(F2*pos_shape, nb_classes, max_norm=regRate)
        #
        self.init_weights()

    def forward(self, x):
        x = self._reshape_input(x)     
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))   
        x = F.elu(x)
        x = self.drop1(self.pool1(x))        
        x = self.conv_sep_point(self.conv_sep_depth(x))        
        x = self.bn3(x)
        x = F.elu(x)
        x = self.drop2(self.pool2(x))
        #
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1) # BCHW->BNC (N=H*W)
        x = self.attn(self.ln1(x), H, W, self.relative_pos)  
        x = self.attn2(self.ln2(x), H, W, self.relative_pos2)
        #
        x = flatten(x, 1)        
        x = self.dense(x)
        return x


class EEGNat4(TorchModel):

    def __init__(self, nb_classes, Chans=64, Samples=128, 
                 F1=4, D=2, kernLength=64, heads=1,
                 dropout=0.5, device="cuda", name="EEGNat"):

        super().__init__(device=device, name=name)
        regRate = .25
        F2 = F1*D
        # EEGNET
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), bias=False, padding='same')
        self.bn1   = nn.BatchNorm2d(F1)
        self.conv2 = Conv2dWithConstraint(F1, F1 * D, (Chans, 1), max_norm=1, bias=False, groups=F1, padding="valid")      
        self.bn2   = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(p=dropout)
        # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
        self.conv_sep_depth = nn.Conv2d(F2, F2, (1, 16), bias=False, groups=F2, padding="same")
        self.conv_sep_point = nn.Conv2d(F2, F2, (1, 1), bias=False, padding="valid")
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(p=dropout)
        # attention
        pos_shape = Samples // 32

        self.ln1   = nn.LayerNorm(F2)        
        self.relative_pos1 = nn.Parameter(torch.randn(heads, pos_shape, pos_shape)) 
        self.attn1   = Attention(dim=F2, num_heads=heads, qkv_bias=False, 
                                      qk_scale=None, attn_drop=0., proj_drop=0., 
                                      qk_ratio=1, sr_ratio=1)
        self.ln2   = nn.LayerNorm(F2)
        self.mlp   = Mlp(F2) 
        #
        self.ln3   = nn.LayerNorm(F2)
        self.relative_pos2 = nn.Parameter(torch.randn(heads, pos_shape, pos_shape)) 
        self.attn2   = Attention(dim=F2, num_heads=heads, qkv_bias=False, 
                                      qk_scale=None, attn_drop=0., proj_drop=0., 
                                      qk_ratio=1, sr_ratio=1) 
        self.ln4   = nn.LayerNorm(F2)
        self.mlp2  = Mlp(F2)     
        #
        self.dense = LineardWithConstraint(F2*pos_shape, nb_classes, max_norm=regRate)
        #
        self.init_weights()

    def forward(self, x):
        x = self._reshape_input(x)     
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))   
        x = F.elu(x)
        x = self.drop1(self.pool1(x))        
        x = self.conv_sep_point(self.conv_sep_depth(x))        
        x = self.bn3(x)
        x = F.elu(x)
        x = self.drop2(self.pool2(x))
        #
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1) # BCHW->BNC (N=H*W)
        x = self.attn1(self.ln1(x), H, W, self.relative_pos1)
        x = self.mlp(self.ln2(x), H, W)
        
        x = self.attn2(self.ln3(x), H, W, self.relative_pos2) 
        x = self.mlp2(self.ln2(x), H, W)    
        x = flatten(x, 1)        
        x = self.dense(x)
        return x
        