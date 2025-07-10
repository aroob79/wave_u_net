import tensorflow as tf
from tensorflow.keras import layers, Model


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.Sequential([
            layers.Conv1D(out_channels, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(out_channels, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

    def call(self, inputs):
        return self.conv(inputs)


class DownBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(out_channels)
        self.pool = layers.MaxPooling1D(pool_size=2)

    def call(self, inputs):
        x = self.conv(inputs)
        x_pooled = self.pool(x)
        return x_pooled, x


class UpBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upconv = layers.Conv1DTranspose(in_channels, 2, strides=2)
        self.conv = ConvBlock( out_channels)

    def call(self, inputs, skip):
        x = self.upconv(inputs)
        x = tf.concat([x, skip], axis=-1)
        return self.conv(x)


class MultiheadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadSelfAttention, self).__init__()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

    def call(self, inputs):
        # TensorFlow MultiHeadAttention expects (batch, seq_len, channels)
        return self.attn(inputs, inputs)


class UNet1D(Model):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128], num_heads=4):
        super(UNet1D, self).__init__()

        self.downs = []
        for i in range(len(features)):
            in_c = in_channels if i == 0 else features[i - 1]
            self.downs.append(DownBlock(features[i]))

        self.attention = MultiheadSelfAttention(features[-1], num_heads)

        self.ups = []
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(UpBlock(features[i], features[i - 1]))

        self.upconv = layers.Conv1DTranspose(features[0], 2, strides=2)
        self.final_conv = layers.Conv1D(out_channels, 1)

    def call(self, inputs):
        skip_connections = []

        # Encoder
        x = inputs

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
        # Apply permutation (swap axes)
        x = tf.transpose(x, perm=[0, 2, 1])  # Change shape (batch, time, channels)
        return x


# Example Usage
if __name__ == "__main__":
    model = UNet1D(in_channels=1, out_channels=4)
    

    # Example Input (batch_size=4, channels=1, sequence_length=1024)
    x = tf.random.normal((11, 1024, 1))
    output = model(x)
    model.summary()
    print(f"Output shape: {output.shape}")
