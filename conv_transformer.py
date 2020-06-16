#%%
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer,
    Dense,
    Dropout,
    LayerNormalization,
    Conv1D,
    Flatten,
)

batch_size = 64

# random seed 설정에 따라 학습이 아예 안되는 경우도 존재한다.
# 5번의 경우 어느정도 잘 되며 random 하게 했을 떄 더 좋은 경우도 있었다.
tf.random.set_seed(5)


class EncoderConvBlock(Layer):
    def __init__(self, m_dim=128):
        super(EncoderConvBlock, self).__init__(name="Encoder_conv_block")
        self.conv1 = Conv1D(
            filters=64, kernel_size=3, padding="same", input_shape=(-1, 10, 30)
        )
        self.layerNorm1 = LayerNormalization(axis=-1)
        self.conv2 = Conv1D(filters=m_dim, kernel_size=3, padding="same")
        self.layerNorm2 = LayerNormalization(axis=-1)

    def call(self, x):
        x = self.conv1(x)
        x = self.layerNorm1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.layerNorm2(x)
        x = tf.nn.relu(x)
        return x


class DecoderConvBlock(Layer):
    def __init__(self, m_dim=128):
        super(DecoderConvBlock, self).__init__(name="Decoder_conv_block")
        self.conv1 = Conv1D(
            filters=64, kernel_size=3, padding="same", input_shape=(-1, 10, 20)
        )
        self.layerNorm1 = LayerNormalization(axis=-1)
        self.conv2 = Conv1D(filters=m_dim, kernel_size=3, padding="same")
        self.layerNorm2 = LayerNormalization(axis=-1)

    def call(self, x):
        x = self.conv1(x)
        x = self.layerNorm1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.layerNorm2(x)
        x = tf.nn.relu(x)
        return x


# TODO: Transformer를 사용할 때, Encoder와 Decoder를 따로 써야하므로 데이터를 SHARP와 XRS 데이터로 쪼개서 사용해야 한다.
class SelfAttention(Layer):
    # m_dim 은 Conv Block에서의 마지막 Conv Channel 수와 같아야 한다.
    def __init__(self, m_dim=128):
        super(SelfAttention, self).__init__(name="self_attention_layer")
        self.m_dim = tf.cast(m_dim, tf.float32)

    def call(self, q, k, v, training=None, mask=None):  # (q,k,v)
        k_T = tf.transpose(k, perm=[0, 1, 3, 2])  # batch,num_heads, depth, seq_len_k
        comp = tf.divide(
            tf.matmul(q, k_T), tf.math.sqrt(self.m_dim)
        )  # batch,num_heads,seq_len_q,seq_len_k
        attention_weights = tf.nn.softmax(
            comp, axis=-1
        )  # batch,num_heads,seq_len_q,seq_len_k
        outputs = tf.matmul(attention_weights, v)  # batch,num_heads, seq_len_q, depth
        return outputs, attention_weights


class MultiHeadAttention(Layer):
    def __init__(self, input_shape, num_heads=8, d_model=128):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, input_shape=input_shape)
        self.wk = tf.keras.layers.Dense(d_model, input_shape=input_shape)
        self.wv = tf.keras.layers.Dense(d_model, input_shape=input_shape)

        self.selfAttn = SelfAttention(m_dim=d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v):
        batch_size = tf.cast(tf.shape(q)[0], tf.float32)
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.selfAttn(q, k, v)
        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


class TransformerBlock(Layer):
    def __init__(self, input_shape, num_heads=8, m_dim=128):
        super(TransformerBlock, self).__init__(name="Transformer_Block")
        self.multiAttn = MultiHeadAttention(
            input_shape=input_shape, num_heads=num_heads, d_model=m_dim
        )
        self.dropout1 = Dropout(0.3)
        # TODO : axis=-1로 하고 SHARP를 넣어줄 때는 conv 에서 shape을 잘 맞춰줘서 넣어야 한다.
        self.layerNorm1 = LayerNormalization(axis=-1)
        self.dense1 = Dense(m_dim, activation="relu",)
        self.dense2 = Dense(m_dim)
        self.dropout2 = Dropout(0.3)
        self.layerNorm2 = LayerNormalization(axis=-1)

    def call(self, query, key, value, training=False):
        # x shape = (batch,10,m_dim)
        context_v, attention_weights = self.multiAttn(
            query, key, value, training=training
        )
        dropout1 = self.dropout1(context_v, training=training)
        residual_con1 = dropout1 + value
        ln1 = self.layerNorm1(residual_con1)
        dense1 = self.dense1(ln1)
        dense2 = self.dense2(dense1)
        dropout2 = self.dropout2(dense2, training=training)
        residual_con2 = ln1 + dropout2
        output = self.layerNorm2(residual_con2)  # (batch,10,m_dim)
        return output


class ConvTransformer(tf.keras.Model):
    def __init__(self, num_heads=8, m_dim=128):
        super(ConvTransformer, self).__init__(name="Conv_Transformer")
        # self.layerNorm = LayerNormalization(axis=1, input_shape=(-1, 10, 50))
        self.encConvBlock = EncoderConvBlock(m_dim=m_dim)
        self.encTransBlock = TransformerBlock(
            input_shape=(-1, 10, m_dim), num_heads=num_heads, m_dim=m_dim
        )
        self.decConvBlock = DecoderConvBlock(m_dim=m_dim)
        self.decTransBlock = TransformerBlock(
            input_shape=(-1, 10, m_dim), num_heads=num_heads, m_dim=m_dim
        )
        self.flat = Flatten()
        self.dense1 = Dense(128, activation="relu",)
        self.dense2 = Dense(3, activation="softmax")

    def call(self, inputs, training=False, mask=None):
        # inputs = self.layerNorm(inputs)
        xrs, sharp = tf.split(inputs, [20, 30], axis=-1)
        encConvOutput = self.encConvBlock(sharp)
        encOutput = self.encTransBlock(
            encConvOutput, encConvOutput, encConvOutput, training=training
        )
        decConvOutput = self.decConvBlock(xrs)
        decTransOutput = self.decTransBlock(
            encOutput, encOutput, decConvOutput, training=training
        )
        flat = self.flat(decTransOutput)
        dense1 = self.dense1(flat)
        output = self.dense2(dense1)
        return output
