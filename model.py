import torch
import torch.nn as nn
from torch.nn import functional as f


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        ## it will keep the array length same like is inp tensor is (2,input channel,11) thrn output shape can be (2,outchannel,11)
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block with max pooling"""

    ## half the sequence length ex (2,inp channel, lin) the output shape will be (2, output channel , lin /2)
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x_pooled = self.pool(x)
        return x_pooled, x  # Return both for skip connection


class UpBlock(nn.Module):
    """Upsampling block with transposed convolution"""

    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upconv = nn.ConvTranspose1d(
            in_channels, in_channels, kernel_size=2, stride=2
        )  ## it will double the input length keeps the channel same
        self.conv = ConvBlock(in_channels * 2, out_channels)  # Handle doubled channels

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)  # Concatenate along channel dimension

        x = self.conv(x)  # Ensure correct channel size
        return x


class MultiheadSelfAttention(nn.Module):
    """Multihead Self-Attention applied on 1D sequence"""

    def __init__(self, embed_dim, num_heads):
        super(MultiheadSelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to (batch, seq_len, channels) for attention
        x, _ = self.attn(x, x, x)  # Self-attention
        return x.permute(0, 2, 1)  # Change back to (batch, channels, seq_len)


class UNet1D(nn.Module):
    """U-Net with Multihead Self-Attention (Conv1D-based for audio)"""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=[16, 32, 64, 128],
        num_heads=4,
    ):
        super(UNet1D, self).__init__()
        # Encoder (Downsampling)
        self.downs = nn.ModuleList()
        for i in range(len(features)):
            in_c = in_channels if i == 0 else features[i - 1]
            self.downs.append(DownBlock(in_c, features[i]))

        # Bottleneck with Multihead Attention
        self.attention = MultiheadSelfAttention(features[-1], num_heads)

        # Decoder (Upsampling)
        self.ups = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(UpBlock(features[i], features[i - 1]))

        self.upconv = nn.ConvTranspose1d(
            features[0], features[0], kernel_size=2, stride=2
        )

        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x, skip = down(x)
            skip_connections.append(skip)

        # Bottleneck with Attention
        x = self.attention(x)

        # Decoder
        for up in self.ups:
            skip = skip_connections.pop()
            x = up(x, skip)
        x = self.upconv(x)
        x = self.final_conv(x)
        return x


# Example Usage
if __name__ == "__main__":
    model = UNet1D(in_channels=1, out_channels=4)
    print(model)

    # Example Input (batch_size=4, channels=1, sequence_length=1024)
    x = torch.randn(11, 1, 1024)
    output = model(x)
    print(f"Output shape: {output.shape}")
