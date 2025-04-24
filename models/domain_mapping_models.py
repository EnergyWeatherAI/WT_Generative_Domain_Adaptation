import torch
import torch.nn as nn


# ------- GENERATOR / MAPPING NETWORK ARCHITECTURE ------
# based on residual blocks of TCNs (see Appendix B of our article)


# the single TCN residual block definition, see GeneratorTCN for overall generator architecture
class Residual_TCN_Block(nn.Module):

    def __init__(self, in_ch, feature_maps, kernel, dilation, added_norm = False):
        super().__init__() 
        self.in_ch = in_ch
        self.ch_out = feature_maps
        self.k = kernel  
        self.dil = dilation
        self.act = nn.Mish() 

        self.pad = int(self.dil * (self.k - 1))

        # identity dimensionality matching 
        self.match_dim = self.in_ch != self.ch_out 
        if self.match_dim:
            self.dim_conv = nn.Conv1d(in_channels=self.in_ch, out_channels = self.ch_out, bias=False, kernel_size=1, stride=1)

        if added_norm is False: 
            tcn_block = [
                nn.Conv1d(self.in_ch, self.ch_out, self.k, padding="same", dilation=self.dil, padding_mode="reflect"),
                nn.Mish(),
                nn.Conv1d(self.ch_out, self.ch_out, self.k, padding="same", dilation=self.dil, padding_mode="reflect"),
                nn.Mish()
            ] 

        else:  # adding a normalization layer after the convolutions, therefore no bias in conv layers
            tcn_block = [
                nn.Conv1d(self.in_ch, self.ch_out, self.k, padding="same", dilation=self.dil, bias=False, padding_mode="reflect"),
                nn.Mish(),
                nn.Conv1d(self.ch_out, self.ch_out, self.k, padding="same", dilation=self.dil, bias=False, padding_mode="reflect"),
                nn.Mish(),
                nn.GroupNorm(1, self.ch_out, affine=False)
            ] 
    

        self.tcn_block = nn.Sequential(*tcn_block)
  


    def forward(self, x):
        tcn = self.tcn_block(x)
        identity = self.dim_conv(x) if self.match_dim else x
        return tcn + identity

        
class GeneratorTCN(nn.Module):

    def __init__(self):
        super().__init__()

        norm = True
        self.block1 = Residual_TCN_Block(in_ch=11, feature_maps = 64, kernel=3, dilation=1, added_norm=False)
        self.block2 = Residual_TCN_Block(in_ch=64, feature_maps = 64, kernel=3, dilation=2, added_norm=norm)
        self.block3 = Residual_TCN_Block(in_ch=64, feature_maps = 64, kernel=3, dilation=4, added_norm=norm)
        self.block4 = Residual_TCN_Block(in_ch=64, feature_maps = 64, kernel=3, dilation=8, added_norm=norm)
        self.block5 = Residual_TCN_Block(in_ch=64, feature_maps = 32, kernel=3, dilation=16, added_norm=norm)
        self.block6 = Residual_TCN_Block(in_ch=32, feature_maps = 16, kernel=3, dilation=32, added_norm=False)

        # ---- OUTPUT CONV ------
        self.conv_out = torch.nn.Sequential(
                        nn.Conv1d(in_channels=16, out_channels = 11, bias=True, kernel_size=1, stride=1))

    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x) 
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.conv_out(x)
        return x 



##### DISCRIMINATOR / CRITIC ARCHITECTURE #####
# based on 1D-cnn blocks with strided convolutions and a final output MLP 

class DiscriminatorAE(nn.Module):
    def __init__(self):
        super().__init__()

        # ----- ENCODE ------
        # block 1
        self.block1 = nn.Sequential(
                                        nn.Conv1d(11, 128, bias=False, kernel_size=5, stride=2, padding=2, padding_mode="reflect"),
                                        nn.Mish(),
                                        nn.GroupNorm(1, 128)
        ) # -> 36


        self.block2 = nn.Sequential(
                                        nn.Conv1d(128, 128, bias=False, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
                                        nn.Mish(),
                                        nn.GroupNorm(1, 128)
        ) # -> 18



        self.block3 = nn.Sequential(
                                        nn.Conv1d(128, 256, bias=False, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
                                        nn.Mish(),
                                        nn.GroupNorm(1, 256)
        ) # -> 9

        self.out = nn.Sequential(nn.Flatten(), nn.Linear(9 * 256, 1))


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.out(x)
        return x