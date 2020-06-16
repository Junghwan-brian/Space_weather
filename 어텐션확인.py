#%%
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    LayerNormalization,
    LSTM,
    Layer,
    Bidirectional,
    Dense,
    Dropout,
)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%%
b_peak = np.load(open("MinMax/train/B_peak.npy", "rb"))  # 2746,20(day,hour),7
b_pro = np.load(open("MinMax/train/B_pro.npy", "rb"))  # (2746, 20, 3)
b_xrs = np.load(open("MinMax/train/B_xrs.npy", "rb"))  # (2746, 20, 5)
a_peak = np.load(open("MinMax/train/A_peak.npy", "rb"))  # 2746,20,7
a_pro = np.load(open("MinMax/train/A_pro.npy", "rb"))  # (2746, 20, 3)
a_xrs = np.load(open("MinMax/train/A_xrs.npy", "rb"))  # (2746, 20, 5)
sharp = np.load(open("MinMax/train/sharp.npy", "rb"))  # (2746, 10, 30)
labels = np.load(open("MinMax/train/labels.npy", "rb")).reshape(-1, 1)  # 2746,1
total_b = np.concatenate(
    [
        b_peak[:, :10, :],  # day
        b_pro[:, :10, :],
        b_xrs[:, :10, :],
        b_peak[:, 10:, :],  # hour
        b_pro[:, 10:, :],
        b_xrs[:, 10:, :],
    ],
    axis=-1,
)  # n,10,30
total_a = np.concatenate(
    [
        a_peak[:, :10, :],  # day
        a_pro[:, :10, :],
        a_xrs[:, :10, :],
        a_peak[:, 10:, :],  # hour
        a_pro[:, 10:, :],
        a_xrs[:, 10:, :],
    ],
    axis=-1,
)  # n,10,30
total_var = np.concatenate([total_b, total_a, sharp], axis=-1)  # n,10,90
train_ds = (
    tf.data.Dataset.from_tensor_slices((total_var, labels.reshape(-1)))
    .batch(len(labels))
    .prefetch(512)
    .shuffle(2500)
)

b_peak = np.load(open("MinMax/test/B_peak.npy", "rb"))  # 596,20(day,hour),7
b_pro = np.load(open("MinMax/test/B_pro.npy", "rb"))  # (596, 20, 3)
b_xrs = np.load(open("MinMax/test/B_xrs.npy", "rb"))  # (596, 20, 5)
a_peak = np.load(open("MinMax/test/A_peak.npy", "rb"))  # 596,20,7
a_pro = np.load(open("MinMax/test/A_pro.npy", "rb"))  # (596, 20, 3)
a_xrs = np.load(open("MinMax/test/A_xrs.npy", "rb"))  # (596, 20, 5)
sharp = np.load(open("MinMax/test/sharp.npy", "rb"))  # (596, 10, 30)
labels = np.load(open("MinMax/test/labels.npy", "rb")).reshape(-1, 1)  # 596,1
total_b = np.concatenate(
    [
        b_peak[:, :10, :],  # day
        b_pro[:, :10, :],
        b_xrs[:, :10, :],
        b_peak[:, 10:, :],  # hour
        b_pro[:, 10:, :],
        b_xrs[:, 10:, :],
    ],
    axis=-1,
)  # n,10,30
total_a = np.concatenate(
    [
        a_peak[:, :10, :],  # day
        a_pro[:, :10, :],
        a_xrs[:, :10, :],
        a_peak[:, 10:, :],  # hour
        a_pro[:, 10:, :],
        a_xrs[:, 10:, :],
    ],
    axis=-1,
)  # n,10,30

total_var = np.concatenate([total_b, total_a, sharp], axis=-1)  # n,10,90

test_ds = (
    tf.data.Dataset.from_tensor_slices((total_var, labels.reshape(-1)))
    .batch(len(labels))
    .prefetch(512)
)


#%%

# 특정 클래스에 대한 정밀도
def single_class_precision(interesting_class_id):
    def prec(y_true, y_pred):
        class_id_true = tf.cast(y_true, tf.int64)
        class_id_pred = tf.math.argmax(y_pred, axis=-1)
        # mask 는 모델의 예측과 보고자 하는 클래스 아이디가 같은 것을 1로 해서 저장함
        # 틀리면 0이 됨. -> 양성이라고 판정한 수
        precision_mask = tf.cast(
            tf.math.equal(class_id_pred, interesting_class_id), "int32"
        )
        # tensor 는 모델이 맞춘 것들과 예측하는 것이 일치하고
        # id 가 예측하고자 하는 것과 같은 것을 저장함. => 즉, 실제 양성수
        class_prec_tensor = (
            tf.cast(tf.math.equal(class_id_true, class_id_pred), "int32")
            * precision_mask
        )
        # 실제 양성수 / 양성이라고 판정한 수
        class_prec = tf.cast(
            tf.math.reduce_sum(class_prec_tensor), "float32"
        ) / tf.cast(tf.math.maximum(tf.math.reduce_sum(precision_mask), 1), "float32")
        return class_prec

    return prec


# 특정 클래스에 대한 재현율
def single_class_recall(interesting_class_id):
    def recall(y_true, y_pred):
        class_id_true = tf.cast(y_true, tf.int64)
        class_id_pred = tf.math.argmax(y_pred, axis=-1)
        # 전체 양성수
        recall_mask = tf.cast(
            tf.math.equal(class_id_true, interesting_class_id), "int32"
        )
        # 검출 양성수
        class_recall_tensor = (
            tf.cast(tf.math.equal(class_id_true, class_id_pred), "int32") * recall_mask
        )
        # 검출 양성수/전체 양성수
        class_recall = tf.cast(
            tf.math.reduce_sum(class_recall_tensor), "float32"
        ) / tf.cast(tf.math.maximum(tf.math.reduce_sum(recall_mask), 1), "float32")
        return class_recall

    return recall


class SharpAttention(Layer):
    def __init__(self, units):
        super(SharpAttention, self).__init__(name="SharpAttention")
        self.w1 = Dense(units)
        self.w2 = Dense(units)
        self.v = Dense(1)

    def call(self, values, query):
        # values = (batch,6,512) , query = (batch,512)
        query = tf.expand_dims(query, 1)
        score = tf.nn.tanh(self.w1(values) + self.w2(query))  # (batch,6,units)
        score = self.v(score)  # (batch,6,1)
        # (batch,6,1) 각 sharp 에 대한 softmax 확률값이 출력됨.
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values  # (batch,6,512)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # batch,512
        return attention_weights, context_vector


class SharpInputModel(tf.keras.Model):
    def __init__(self, units):
        super(SharpInputModel, self).__init__(name="SharpInputModel")
        self.layerNorm = LayerNormalization(axis=-2, input_shape=(-1, 10, 30))
        self.lstm1 = Bidirectional(LSTM(256))
        self.abs_lstm = Bidirectional(LSTM(256, input_shape=(-1, 10, 5)))
        self.sav_lstm = Bidirectional(LSTM(256, input_shape=(-1, 10, 5)))
        self.totp_lstm = Bidirectional(LSTM(256, input_shape=(-1, 10, 5)))
        self.toth_lstm = Bidirectional(LSTM(256, input_shape=(-1, 10, 5)))
        self.totz_lstm = Bidirectional(LSTM(256, input_shape=(-1, 10, 5)))
        self.usf_lstm = Bidirectional(LSTM(256, input_shape=(-1, 10, 5)))
        self.attn = SharpAttention(units)
        self.attention_weigths = None
        self.values = None

    def call(self, inputs):
        inputs = self.layerNorm(inputs)
        query = self.lstm1(inputs)  # batch,512

        abs, sav, totp, toth, totz, usf = tf.split(
            tf.reshape(inputs, (-1, 10, 5, 6)), num_or_size_splits=6, axis=-1
        )  # batch , 10, 5, 1 -> 각 sharp 변수별 데이터로 변형.
        abs = self.abs_lstm(tf.reshape(abs, (-1, 10, 5)))
        sav = self.sav_lstm(tf.reshape(sav, (-1, 10, 5)))
        totp = self.totp_lstm(tf.reshape(totp, (-1, 10, 5)))
        toth = self.toth_lstm(tf.reshape(toth, (-1, 10, 5)))
        totz = self.totz_lstm(tf.reshape(totz, (-1, 10, 5)))
        usf = self.usf_lstm(tf.reshape(usf, (-1, 10, 5)))
        self.values = tf.stack([abs, sav, totp, toth, totz, usf], axis=1)
        # (batch,6,1), (batch,512)
        self.attention_weigths, context_vector = self.attn(self.values, query)
        return context_vector, self.attention_weigths


class bahdanau(Layer):
    def __init__(self, units):
        super(bahdanau, self).__init__(name="attention_layer")
        self.w1 = Dense(units)
        self.w2 = Dense(units)
        self.v = Dense(1)

    def call(self, query, values):
        # lstm features는 양방향이라 실제로는 2배를 해준 값이 사용된다.
        query = tf.expand_dims(query, axis=1)  # (batch,1,lstm features)
        score = tf.nn.tanh(self.w1(values) + self.w2(query))  # (batch,T,units)
        score = self.v(score)  # (batch,T,1)
        # (batch,T,1) 각 time softmax 확률값이 출력됨.
        attention_weights = tf.nn.softmax(score, axis=1)
        # context_vector shape = (batch,T,lstm features)
        context_vector = attention_weights * values
        # axis=1로 더한다는 것은 각 Time step 특징 값을 더한다는 뜻.
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch,lstm features)
        return context_vector, attention_weights


class XrsInputModel(tf.keras.Model):
    def __init__(self):
        super(XrsInputModel, self).__init__(name="XrsInputModel")
        self.layerNorm = LayerNormalization(axis=-2, input_shape=(-1, 10, 30))
        self.lstm = Bidirectional(LSTM(256, return_sequences=True, return_state=True))
        self.attention = bahdanau(128)
        self.attention_weights = None

    def call(self, inputs):
        H, forward_h, forward_c, backward_h, backward_c = self.lstm(
            self.layerNorm(inputs)
        )
        query = tf.concat([forward_h, backward_h], axis=-1)
        context_vector, self.attention_weights = self.attention(query, H)
        return context_vector  # batch, 512


class AttentionModel(tf.keras.Model):
    def __init__(self):
        super(AttentionModel, self).__init__(name="AttentionModel")
        self.lstm = Bidirectional(LSTM(256, input_shape=(-1, 10, 90)))
        self.b_model = XrsInputModel()
        self.a_model = XrsInputModel()
        self.sharp_model = SharpInputModel(128)
        self.dropout = Dropout(0.3)
        self.w1 = Dense(128)
        self.w2 = Dense(128)
        self.v = Dense(1)
        self.attention_weights = None
        self.dense = Dense(4, activation="softmax")

    def call(self, inputs, training=False):
        query = self.lstm(inputs)[:, tf.newaxis, :]  # batch,1,512

        b_input, a_input, sharp = tf.split(inputs, 3, axis=-1)
        b_output = self.b_model(b_input)  # batch,512
        a_output = self.a_model(a_input)  # batch,512
        sharp_output, sharp_attn_weights = self.sharp_model(sharp)  # batch,512

        b_output = b_output[:, tf.newaxis, :]
        a_output = a_output[:, tf.newaxis, :]
        sharp_output = sharp_output[:, tf.newaxis, :]

        values = tf.concat([b_output, a_output, sharp_output], axis=1)  # batch,3,512
        score = tf.nn.tanh(self.w1(query) + self.w2(values))  # batch,3,128
        score = self.v(score)  # batch,3,1
        self.attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = self.attention_weights * values  # batch,3,512
        context_vector = tf.reduce_sum(context_vector, axis=1)
        outputs = self.dropout(context_vector, training=training)
        outputs = self.dense(outputs)
        return outputs, sharp_attn_weights, self.attention_weights


#%%
model = AttentionModel()
model.build(input_shape=(64, 10, 90))
# 4번 모델의 precision : 1.0 , recall : 0.02 -> 굉장히 보수적으로 데이터를 뽑는다.
# 3번 모델의 precision : 0.6 , recall : 0.1
# 2번 모델의 precision : 1.0 , recall : 0.06
model.load_weights("attention_model/ckpt-3")

#%%
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
    train_prec,
    train_rec,
):
    with tf.GradientTape() as tape:
        prediction, sharp_attn_weights, attention_weigths = model(inputs, training=True)
        loss = loss_obj(labels, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, prediction)
    train_prec(precision(labels, prediction))
    train_rec(recall(labels, prediction))


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
    test_prec,
    test_rec,
):
    prediction, sharp_attn_weights, attention_weigths = model(inputs, training=False)
    loss = loss_obj(labels, prediction)
    test_loss(loss)
    test_accuracy(labels, prediction)
    test_prec(precision(labels, prediction))
    test_rec(recall(labels, prediction))


optimizer = tf.keras.optimizers.Adam(0.001)
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
test_loss = tf.keras.metrics.Mean()
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

precision = single_class_precision(0)
recall = single_class_recall(0)

train_prec = tf.keras.metrics.Mean()
test_prec = tf.keras.metrics.Mean()

train_rec = tf.keras.metrics.Mean()
test_rec = tf.keras.metrics.Mean()
train_ac_list = []
train_loss_list = []
train_prec_list = []
train_rec_list = []
test_ac_list = []
test_loss_list = []
test_prec_list = []
test_rec_list = []
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, "./att_ckpts", max_to_keep=3)
compare_value = 0.5
for epoch in range(15):
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
            train_prec,
            train_rec,
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
            test_prec,
            test_rec,
        )
    ckpt.step.assign_add(1)
    val_recall = test_rec.result()
    val_precision = test_prec.result()
    if val_recall + val_precision > compare_value:
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        compare_value = val_recall + val_precision
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
    train_loss_list.append(train_loss.result())
    train_ac_list.append(train_accuracy.result() * 100)
    train_prec_list.append(train_prec.result())
    train_rec_list.append(train_rec.result())

    test_loss_list.append(test_loss.result())
    test_ac_list.append(test_accuracy.result() * 100)
    test_prec_list.append(test_prec.result())
    test_rec_list.append(test_rec.result())

    test_loss.reset_states()
    test_accuracy.reset_states()
    train_loss.reset_states()
    train_accuracy.reset_states()
    train_rec.reset_states()
    train_prec.reset_states()
    test_rec.reset_states()
    test_prec.reset_states()
#%%
precision = single_class_precision(1)
recall = single_class_recall(1)
t_inputs, t_labels = next(iter(test_ds))
pred, sharp_attn_weights, attn_weights = model(t_inputs)
prec = precision(t_labels, pred)
rec = recall(t_labels, pred)
acc = test_accuracy(t_labels, pred)
loss = loss_obj(t_labels, pred)
print(prec, rec)
print(acc, loss)
#%%
plt.figure(figsize=(12, 9))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.xticks(rotation=45,)
    plt.bar(
        ["ABSNJZH", "SAVNCPP", "TOTPOT", "TOTUSJH", "TOTUSJZ", "USFLUX"],
        sharp_attn_weights[10 + i][:, 0],
    )
    if i < 2:
        plt.title("X_M RANK")
    elif i == 2:
        plt.title("C RANK")
    else:
        plt.title("B RANK")
plt.show()

#%%
df = pd.DataFrame(
    np.array(sharp_attn_weights[:, :, 0]),
    columns=["ABSNJZH", "SAVNCPP", "TOTPOT", "TOTUSJH", "TOTUSJZ", "USFLUX"],
    index=[i for i in range(len(t_labels))],
)
df["labels"] = t_labels
print(np.mean(df.iloc[:, 0]))
print(np.mean(df.iloc[:, 1]))
print(np.mean(df.iloc[:, 2]))
print(np.mean(df.iloc[:, 3]))
print(np.mean(df.iloc[:, 4]))
print(np.mean(df.iloc[:, 5]))
#%%
plt.figure(figsize=(8, 10))
for i in range(6):
    plt.subplot(3, 2, i + 1)
    sns.boxplot(data=df, y=df.columns[i])
plt.show()
#%%
df_mean = df.groupby("labels").mean()
plt.figure(figsize=(12, 9))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.bar(df_mean.columns, df_mean.iloc[i])
    if i == 0:
        plt.title("X_M RANK")
    elif i == 1:
        plt.title("C RANK")
    elif i == 2:
        plt.title("B RANK")
    else:
        plt.title("None Rank")
plt.show()
#%%
attn_weights_arr = np.array(attn_weights)[:, :, 0]
plt.figure(figsize=(12, 9))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.xticks(rotation=45,)
    plt.bar(
        ["B_XRS", "A_XRS", "SHARP"], attn_weights_arr[6 + i, :],
    )
    if i < 2:
        plt.title("X_M RANK")
    elif i == 2:
        plt.title("C RANK")
    else:
        plt.title("B RANK")
plt.show()
#%%
attn_weights_arr = np.array(attn_weights)[:, :, 0]
df = pd.DataFrame(
    attn_weights_arr,
    columns=["B_XRS", "A_XRS", "SHARP"],
    index=[i for i in range(596)],
)
df["labels"] = t_labels
df_mean = df.groupby("labels").mean()
plt.figure(figsize=(12, 9))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.bar(df_mean.columns, df_mean.iloc[i])
    if i == 0:
        plt.title("X_M RANK")
    elif i == 1:
        plt.title("C RANK")
    elif i == 2:
        plt.title("B RANK")
    else:
        plt.title("None Rank")
plt.show()
#%%
b_peak = np.load(open("MinMax/train/B_peak.npy", "rb"))  # 2746,20(day,hour),7
b_pro = np.load(open("MinMax/train/B_pro.npy", "rb"))  # (2746, 20, 3)
b_xrs = np.load(open("MinMax/train/B_xrs.npy", "rb"))  # (2746, 20, 5)
a_peak = np.load(open("MinMax/train/A_peak.npy", "rb"))  # 2746,20,7
a_pro = np.load(open("MinMax/train/A_pro.npy", "rb"))  # (2746, 20, 3)
a_xrs = np.load(open("MinMax/train/A_xrs.npy", "rb"))  # (2746, 20, 5)
sharp = np.load(open("MinMax/train/sharp.npy", "rb"))  # (2746, 10, 30)
labels = np.load(open("MinMax/train/labels.npy", "rb")).reshape(-1, 1)  # 2746,1
i = 0
abs = np.concatenate(
    [
        sharp[:, :, i],
        sharp[:, :, i + 6],
        sharp[:, :, i + 12],
        sharp[:, :, i + 18],
        sharp[:, :, i + 24],
    ],
    axis=-1,
)
sav = np.concatenate(
    [
        sharp[:, :, i + 1],
        sharp[:, :, i + 1 + 6],
        sharp[:, :, i + 1 + 12],
        sharp[:, :, i + 1 + 18],
        sharp[:, :, i + 1 + 24],
    ],
    axis=-1,
)
totp = np.concatenate(
    [
        sharp[:, :, i + 2],
        sharp[:, :, i + 2 + 6],
        sharp[:, :, i + 2 + 12],
        sharp[:, :, i + 2 + 18],
        sharp[:, :, i + 2 + 24],
    ],
    axis=-1,
)
toth = np.concatenate(
    [
        sharp[:, :, i + 3],
        sharp[:, :, i + 3 + 6],
        sharp[:, :, i + 3 + 12],
        sharp[:, :, i + 3 + 18],
        sharp[:, :, i + 3 + 24],
    ],
    axis=-1,
)
totz = np.concatenate(
    [
        sharp[:, :, i + 4],
        sharp[:, :, i + 4 + 6],
        sharp[:, :, i + 4 + 12],
        sharp[:, :, i + 4 + 18],
        sharp[:, :, i + 4 + 24],
    ],
    axis=-1,
)
usf = np.concatenate(
    [
        sharp[:, :, i + 5],
        sharp[:, :, i + 5 + 5],
        sharp[:, :, i + 5 + 12],
        sharp[:, :, i + 5 + 18],
        sharp[:, :, i + 5 + 24],
    ],
    axis=-1,
)
sharp_name_list = ["ABSNJZH", "SAVNCPP", "TOTPOT", "TOTUSJH", "TOTUSJZ", "USFLUX"]
sharp_data = [abs, sav, totp, toth, totz, usf]
#%%
for i in range(6):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(y=sharp_data[i].reshape(-1))
    plt.ylim(0, 1)
    plt.title(sharp_name_list[i])
plt.show()
