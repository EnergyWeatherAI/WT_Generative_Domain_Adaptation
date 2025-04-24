import torch.nn as nn 

###-------- NBM MODEL ARCHITECTURE ---------
# The autoencoder architecture for the NBMs, see Appendix B.


class base_AE_CNN(nn.Module):        
    def __init__(self, in_channels = 11):
        # input samples for this conv1d model was batch_size x channels (11) x length (72) in our work.
        super().__init__()
        self.in_ch = in_channels
        # ----- ENCODER ------
        self.conv_block_1 = nn.Sequential(
                        nn.Conv1d(in_channels = self.in_ch, out_channels = 32, bias=True, kernel_size = 7, stride=1, padding="same", padding_mode="reflect"),
                        nn.Mish(),
                        nn.Conv1d(in_channels = 32, out_channels = 32, bias=False, kernel_size = 7, stride=1, padding="same", padding_mode="reflect"),
                        nn.Mish(),
                        nn.MaxPool1d(2),
                        nn.GroupNorm(1, 32)
                        )

        self.conv_block_2 = nn.Sequential(
                        nn.Conv1d(in_channels = 32, out_channels = 32, bias=True, kernel_size = 5, stride=1, padding="same", padding_mode="reflect"),
                        nn.Mish(),
                        nn.Conv1d(in_channels = 32, out_channels = 32, bias=False, kernel_size = 5, stride=1, padding="same", padding_mode="reflect"),
                        nn.Mish(),
                        nn.MaxPool1d(2),
                        nn.GroupNorm(1, 32)
                        )
        
        # ----- MLP BOTTLENECK -----
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
                                    nn.Linear(32 * 18, 72)) 

        # ----- DECODER ------
        self.decoder1 = nn.Linear(72, 18 * 8)
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_block_3 = nn.Sequential(
                        nn.Conv1d(in_channels = 8, out_channels = 32, bias=True, kernel_size = 3, stride=1, padding="same", padding_mode="reflect"),
                        nn.Mish(),
                        nn.Conv1d(in_channels = 32, out_channels = 32, bias=False, kernel_size = 3, stride=1, padding="same", padding_mode="reflect"),
                        nn.Mish(),
                        nn.GroupNorm(1, 32)
                        )

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_block_4 = nn.Sequential(
                        nn.Conv1d(in_channels = 32, out_channels = 32, bias=True, kernel_size = 3, stride=1, padding="same", padding_mode="reflect"),
                        nn.Mish(),
                        nn.Conv1d(in_channels = 32, out_channels = 32, bias=False, kernel_size = 3, stride=1, padding="same", padding_mode="reflect"),
                        nn.Mish(),
                        nn.GroupNorm(1, 32)
                        )


        self.conv_out = nn.Sequential(
                            nn.Conv1d(in_channels=32, out_channels = self.in_ch, 
                                kernel_size=1, stride=1, bias=True, padding="same", padding_mode="reflect"))

    def forward(self, x):
        x = self.conv_block_1(x) # in: 11ch x 72l
        x = self.conv_block_2(x)
        x = self.flatten(x)
        x = self.encoder(x) # encoded feature representation
        x = self.decoder1(x)
        x = x.view(-1, 8, 18)
        x = self.upsample_1(x)
        x = self.conv_block_3(x)
        x = self.upsample_2(x)
        x = self.conv_block_4(x)
        x = self.conv_out(x) #out: 11ch x 72l
        return x