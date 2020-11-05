#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.basic_lstm import basic_lstm
from batchData import GetData
from metric.precision_recall import *

model = basic_lstm()

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

ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, "./att_ckpts", max_to_keep=3)
compare_value = 0.5
data = GetData()
train_indexes, val_indexes = data.kfold_indexes(n_split=5)
train_ac_list = []
train_loss_list = []
train_prec_list = []
train_rec_list = []
test_ac_list = []
test_loss_list = []
test_prec_list = []
test_rec_list = []
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
            prediction = model(inputs, training=True)
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
        prediction = model(inputs, training=False)
        loss = loss_obj(labels, prediction)
        test_loss(loss)
        test_accuracy(labels, prediction)
        test_prec(precision(labels, prediction))
        test_rec(recall(labels, prediction))

    for epoch in range(1):
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
        test_rec.reset_states()
        test_prec.reset_states()

        train_loss.reset_states()
        train_accuracy.reset_states()
        train_rec.reset_states()
        train_prec.reset_states()
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
model.load_weights("att_ckpts/ckpt-3")

#%%
a, b = next(iter(val_ds))
pred = model.predict(a)
print("recall: ", single_class_recall(1)(b, pred).numpy())
print("precision: ", single_class_precision(1)(b, pred).numpy())
#%%
sharp = np.array(a)[:, :, -30:]
label = np.array(b)
zero_count = 0
for i in range(275):
    if label[i] == 0 and np.min(sharp[i]) == 0:
        print(np.max(sharp[i]))
        zero_count += 1
