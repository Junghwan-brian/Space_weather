#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from metric.precision_recall import *
from batchData import GetData
from models.attention_model import AttentionModel
import pandas as pd

train_ds, test_ds = GetData().get_train_test_ds()


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


model = AttentionModel()
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, "./att_ckpts", max_to_keep=3)

compare_value = 0.5
#%%
for epoch in range(20):
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
    if 2 * (val_recall * val_precision) / (val_recall + val_precision) > compare_value:
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        compare_value = 2 * (val_recall * val_precision) / (val_recall + val_precision)
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
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(test_ac_list, label="test")
plt.plot(train_ac_list, label="train")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(test_loss_list, label="test")
plt.plot(train_loss_list, label="train")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(test_prec_list, label="test")
plt.plot(train_prec_list, label="train")
plt.title("Precision")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(test_rec_list, label="test")
plt.plot(train_rec_list, label="train")
plt.title("Recall")
plt.xlabel("Epoch")
plt.legend()

plt.show()
#%%
"""
각 SHARP 데이터별 어텐션 벨류를 시각화한다.
"""
test_data, test_labels = next(iter(test_ds))
pred, sharp_attn_weights, attn_weights = model(test_data, training=False)
test_recall = recall(test_labels, pred)
test_precision = precision(test_labels, pred)
fscore = 2 * (test_recall * test_precision) / (test_recall + test_precision)
print(fscore)
print(test_recall)
print(test_precision)
#%%
df = pd.DataFrame(
    np.array(sharp_attn_weights[:, :, 0]),
    columns=["ABSNJZH", "SAVNCPP", "TOTPOT", "TOTUSJH", "TOTUSJZ", "USFLUX"],
    index=[i for i in range(len(test_labels))],
)
df["labels"] = test_labels
df_mean = df.groupby("labels").mean()
plt.figure(figsize=(12, 9))
for i in range(3):
    plt.subplot(2, 2, i + 1)
    plt.bar(df_mean.columns, df_mean.iloc[i])
    if i == 0:
        plt.title("X_M RANK")
    elif i == 1:
        plt.title("C RANK")
    elif i == 2:
        plt.title("None Rank")
plt.show()
#%%
"""
XRS, SHARP 어텐션 벨류를 시각화한다.
"""
attn_weights_arr = np.array(attn_weights)[:, :, 0]
df = pd.DataFrame(
    attn_weights_arr,
    columns=["B_XRS", "SHARP"],
    index=[i for i in range(len(test_labels))],
)
df["labels"] = test_labels
df_mean = df.groupby("labels").mean()
plt.figure(figsize=(12, 9))
for i in range(3):
    plt.subplot(2, 2, i + 1)
    plt.bar(df_mean.columns, df_mean.iloc[i])
    if i == 0:
        plt.title("X_M RANK")
    elif i == 1:
        plt.title("C RANK")
    elif i == 2:
        plt.title("None Rank")
plt.show()
