#%%
import tensorflow as tf
from batchData import GetData
from metric.precision_recall import *

import matplotlib.pyplot as plt
from models.conv_transformer import ConvTransformer

batch_size = 128
m_dim = 512
model = ConvTransformer(num_heads=16, m_dim=m_dim)
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


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=3000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(m_dim)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-9
)
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

for epoch in range(100):
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
