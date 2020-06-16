#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from attention_model import AttentionModel
from batchData import GetData
from precision_recall import *

model = AttentionModel()
model.build(input_shape=(64, 10, 50))
#%%
# print(model.summary())
# print(model.b_model.summary())
# print(model.sharp_model.summary())
# model.load_weights("attention_model/attention_model")


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
data = GetData()
train_indexes, val_indexes = data.kfold_indexes(n_split=5)
for train_idx, val_idx in zip(train_indexes, val_indexes):
    train_ds, val_ds = data.get_kfold_dataset(train_idx, val_idx)

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
            prediction, sharp_attn_weights, attention_weigths = model(
                inputs, training=True
            )
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
        prediction, sharp_attn_weights, attention_weigths = model(
            inputs, training=False
        )
        loss = loss_obj(labels, prediction)
        test_loss(loss)
        test_accuracy(labels, prediction)
        test_prec(precision(labels, prediction))
        test_rec(recall(labels, prediction))

    for epoch in range(30):
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
        for t_inputs, t_labels in val_ds:
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
            f"train_loss : {train_loss.result()} , train_accuracy : {train_accuracy.result()*100}"
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
plt.plot(test_ac_list, label="validation")
plt.plot(train_ac_list, label="train")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(test_loss_list, label="validation")
plt.plot(train_loss_list, label="train")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(test_prec_list, label="validation")
plt.plot(train_prec_list, label="train")
plt.title("Precision")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(test_rec_list, label="validation")
plt.plot(train_rec_list, label="train")
plt.title("Recall")
plt.xlabel("Epoch")
plt.legend()

plt.show()
#%%
i = 0
j = 0
k = 0
for a, b in val_ds:
    pred, sharp_attn_weights, attn_weights = model(a, training=False)
    for idx in range(275):
        if np.argmax(pred[idx]) == 0 and b[idx].numpy() == 0:
            j += 1
        if np.argmax(pred[idx]) == 0:
            i += 1
        if b[idx].numpy() == 0:
            k += 1
print("Precision: ", j / i)
print("recall: ", j / k)
#%%
plt.figure(figsize=(12, 9))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.xticks(rotation=45,)
    plt.bar(
        ["ABSNJZH", "SAVNCPP", "TOTPOT", "TOTUSJH", "TOTUSJZ", "USFLUX"],
        sharp_attn_weights[i + 4][:, 0],
    )
    if i == 0:
        plt.title("None Rank")
    elif i == 1:
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
    index=[i for i in range(550)],
)
df["labels"] = b
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
        plt.title("None Rank")
plt.show()
#%%
attn_weights_arr = np.array(attn_weights)[:, :, 0]
df = pd.DataFrame(
    attn_weights_arr,
    columns=["B_XRS", "A_XRS", "SHARP"],
    index=[i for i in range(550)],
)
df["labels"] = b
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
