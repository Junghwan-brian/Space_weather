#%%
import tensorflow as tf
import numpy as np
from batchData import GetData
from precision_recall import *
from tensorflow.keras.layers import (
    Layer,
    Dense,
    Dropout,
    LayerNormalization,
    Conv1D,
    Flatten,
)

import matplotlib.pyplot as plt

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


model = ConvTransformer(num_heads=8, m_dim=256)
model.build(input_shape=(batch_size, 10, 50))
model.summary()
train_ds, test_ds = GetData().get_train_test_ds(train_batch_size=batch_size)


@tf.function
def train_step(
    model,
    inputs,
    labels,
    optimizer,
    loss_obj,
    train_loss,
    train_accuracy,
    precision,
    recall,
    csi,
    train_prec,
    train_rec,
    train_csi,
):
    with tf.GradientTape() as tape:
        prediction = model(inputs, training=True)
        loss = loss_obj(labels, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, prediction)
    train_prec(precision(labels, prediction))
    train_rec(recall(labels, prediction))
    train_csi(csi(labels, prediction))


@tf.function
def test_step(
    model,
    inputs,
    labels,
    loss_obj,
    test_loss,
    test_accuracy,
    precision,
    recall,
    csi,
    test_prec,
    test_rec,
    test_csi,
):
    prediction = model(inputs, training=False)
    loss = loss_obj(labels, prediction)
    test_loss(loss)
    test_accuracy(labels, prediction)
    test_prec(precision(labels, prediction))
    test_rec(recall(labels, prediction))
    test_csi(csi(labels, prediction))


learning_rate = 0.001
steps_per_epoch = 2746 / batch_size
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=steps_per_epoch * 5,
    decay_rate=0.8,
    staircase=True,
)
optimizer = tf.keras.optimizers.Adam(lr_schedule)
loss_obj = tf.keras.losses.CategoricalCrossentropy()
train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.CategoricalAccuracy()
test_loss = tf.keras.metrics.Mean()
test_accuracy = tf.keras.metrics.CategoricalAccuracy()

precision = single_class_precision(0)
recall = single_class_recall(0)
csi = single_class_csi(0)

train_prec = tf.keras.metrics.Mean()
test_prec = tf.keras.metrics.Mean()

train_rec = tf.keras.metrics.Mean()
test_rec = tf.keras.metrics.Mean()

train_csi = tf.keras.metrics.Mean()
test_csi = tf.keras.metrics.Mean()


train_ac_list = []
train_loss_list = []
train_prec_list = []
train_rec_list = []
train_fscore_list = []
train_csi_list = []


test_ac_list = []
test_loss_list = []
test_prec_list = []
test_rec_list = []
test_fscore_list = []
test_csi_list = []

ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, "./att_ckpts", max_to_keep=3)

compare_value = 0.4

for epoch in range(50):
    for inputs, labels in train_ds:
        train_step(
            model,
            inputs,
            labels,
            optimizer,
            loss_obj,
            train_loss,
            train_accuracy,
            precision,
            recall,
            csi,
            train_prec,
            train_rec,
            train_csi,
        )
    for t_inputs, t_labels in test_ds:
        test_step(
            model,
            t_inputs,
            t_labels,
            loss_obj,
            test_loss,
            test_accuracy,
            precision,
            recall,
            csi,
            test_prec,
            test_rec,
            test_csi,
        )
    ckpt.step.assign_add(1)
    val_recall = test_rec.result()
    val_precision = test_prec.result()
    train_recall = train_rec.result()
    train_precision = train_prec.result()
    if train_recall != 0 or train_precision != 0:
        train_fscore = (
            2 * (train_recall * train_precision) / (train_recall + train_precision)
        )
    else:
        train_fscore = 0
    if val_recall != 0 or val_precision != 0:
        test_fscore = 2 * (val_recall * val_precision) / (val_recall + val_precision)
    else:
        test_fscore = 0
    if test_csi.result() > compare_value:
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        compare_value = test_csi.result()
    print("Epoch : ", epoch + 1)
    print(
        f"train_loss : {train_loss.result()} , train_accuracy : {train_accuracy.result() * 100}"
    )
    print(
        f"train_precision : {train_prec.result()} , train_recall : {train_rec.result()} \n"
    )
    print(
        f"test_loss : {test_loss.result()} , test_accuracy : {test_accuracy.result() * 100}"
    )
    print(
        f"test_precision : {test_prec.result()} , test_recall : {test_rec.result()} \n"
    )
    print(f"train_fscore = {train_fscore} , test_fscore = {test_fscore} \n")
    print(f"train_csi = {train_csi.result()} , test_csi = {test_csi.result()} \n")

    train_loss_list.append(train_loss.result())
    train_ac_list.append(train_accuracy.result() * 100)
    train_prec_list.append(train_prec.result())
    train_rec_list.append(train_rec.result())
    train_fscore_list.append(train_fscore)
    train_csi_list.append(train_csi.result())

    test_loss_list.append(test_loss.result())
    test_ac_list.append(test_accuracy.result() * 100)
    test_prec_list.append(test_prec.result())
    test_rec_list.append(test_rec.result())
    test_fscore_list.append(test_fscore)
    test_csi_list.append(test_csi.result())

    test_loss.reset_states()
    test_accuracy.reset_states()
    test_rec.reset_states()
    test_prec.reset_states()
    test_csi.reset_states()

    train_loss.reset_states()
    train_accuracy.reset_states()
    train_rec.reset_states()
    train_prec.reset_states()
    train_csi.reset_states()
print("MAX CSI: ", compare_value)
#%%
plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(test_ac_list, label="test")
plt.plot(train_ac_list, label="train")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(test_loss_list, label="test")
plt.plot(train_loss_list, label="train")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(test_prec_list, label="test")
plt.plot(train_prec_list, label="train")
plt.title("Precision")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(test_rec_list, label="test")
plt.plot(train_rec_list, label="train")
plt.title("Recall")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(test_fscore_list, label="test")
plt.plot(train_fscore_list, label="train")
plt.title("Fscore")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(test_csi_list, label="test")
plt.plot(train_csi_list, label="train")
plt.title("Critical success index")
plt.xlabel("Epoch")
plt.legend()

plt.show()

#%%
t_inputs, t_labels = next(iter(test_ds))
pred = model.predict(t_inputs)
print(single_class_recall(0)(t_labels, pred))
print(single_class_precision(0)(t_labels, pred))
print(single_class_csi(0)(t_labels, pred))
print(single_class_recall(1)(t_labels, pred))
print(single_class_precision(1)(t_labels, pred))
print(single_class_csi(1)(t_labels, pred))
print(single_class_recall(2)(t_labels, pred))
print(single_class_precision(2)(t_labels, pred))
print(single_class_csi(2)(t_labels, pred))
#%%
from collections import Counter

train_ds, test_ds = GetData().get_train_test_ds(train_batch_size=512)
train_labels = []
for a, b in train_ds:
    for label in b:
        train_labels.append(np.argmax(label))
    break
# labels = []
# for label in t_labels:
#     labels.append(np.argmax(label))

print(Counter(train_labels))
