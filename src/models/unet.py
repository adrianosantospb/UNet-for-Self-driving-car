# DL library imports
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, layer_channels):
        super(UNetEncoder, self).__init__()
        self.encoder = nn.ModuleList()

        for n_channels in layer_channels:
            self.encoder.append(double_conv(in_channels, n_channels))
            in_channels = n_channels 

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        skip_connections = []
        for e in self.encoder:
            x = e(x)
            skip_connections.append(x)
            x = self.pool(x)
        return x, skip_connections
			
			
class UNetDecoder(nn.Module):
    def __init__(self, layer_channels):
        super(UNetDecoder, self).__init__()
        self.decoder = nn.ModuleList()        

        for num_channels in reversed(layer_channels):
            self.decoder.append(nn.ConvTranspose2d(num_channels*2, num_channels, kernel_size=2, stride=2))
            self.decoder.append(double_conv(num_channels*2, num_channels))

    def forward(self, x, skip_connections):
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)

            skip_conn = skip_connections[idx//2]
            if x.shape != skip_conn.shape:
                x = TF.resize(x, size=skip_conn.shape[2:])

            concat_skip = torch.cat((skip_conn, x), dim=1)
            x = self.decoder[idx+1](concat_skip)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, layer_channels):
        super(UNet, self).__init__()
        
        # Encoder and decoder modules
        self.encoder = UNetEncoder(in_channels, out_channels, layer_channels)
        self.decoder = UNetDecoder(layer_channels)

        # conv layer to transition from encoder to decoder and 
        # 1x1 convolution to reduce num channels to out_channels
        self.bottleneck = double_conv(layer_channels[-1], layer_channels[-1]*2)
        self.final_conv = nn.Conv2d(layer_channels[0], out_channels, kernel_size=1)
        
        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)

    def forward(self, x):
        # Encoder blocks
        encoder_output, skip_connections = self.encoder(x)

        # transition between encoder and decoder
        x = self.bottleneck(encoder_output)

        # we need the last skip connection first
        # so reversing the list 
        skip_connections = skip_connections[::-1]

        # Decoder blocks
        x = self.decoder(x, skip_connections)

        # final 1x1 conv to match input size
        return self.final_conv(x)        
