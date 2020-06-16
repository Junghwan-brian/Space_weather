import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

np.random.seed(0)


class GetData:
    def __init__(self, under_sampling_ratio=0):
        self.train_var = np.load(open("train/sharp_xrs.npy", "rb"))  # (2746, 10, 50)
        self.train_labels = np.load(open("train/labels.npy", "rb"))  # 2746,3
        self.test_var = np.load(open("test/sharp_xrs.npy", "rb"))  # (2746, 10, 50)
        self.test_labels = np.load(open("test/labels.npy", "rb"))  # 2746,3
        if under_sampling_ratio != 0:
            self.new_train_var = []
            self.new_train_labels = []
            for var, label in zip(self.train_var, self.train_labels):
                if (
                    np.random.uniform(0, 1) < under_sampling_ratio
                    and np.argmax(label) == 2
                ) or np.argmax(label) != 2:
                    self.new_train_labels.append(label)
                    self.new_train_var.append(var)
            self.train_var = self.new_train_var
            self.train_labels = self.new_train_labels

    def kfold_indexes(self, n_split=5):
        kf = KFold(n_splits=n_split, shuffle=True, random_state=2020)
        train_indexes = []
        val_indexes = []
        for train_idx, val_idx in kf.split(self.train_labels):
            train_indexes.append(train_idx)
            val_indexes.append(val_idx)
        return train_indexes, val_indexes

    def get_kfold_dataset(self, train_idx, val_idx, train_batch_size=64):
        self.train_total = self.train_var[train_idx[0]][np.newaxis, ...]
        train_labels = self.train_labels[train_idx[0]][np.newaxis, ...]
        self.val_total = self.train_var[val_idx[0]][np.newaxis, ...]
        val_labels = self.train_labels[val_idx[0]][np.newaxis, ...]
        for idx in train_idx[1:]:
            self.train_total = np.concatenate(
                [self.train_total, self.train_var[idx][np.newaxis, ...]], axis=0
            )
            train_labels = np.concatenate(
                [train_labels, self.train_labels[idx][np.newaxis, ...]], axis=0
            )
        for idx in val_idx[1:]:
            self.val_total = np.concatenate(
                [self.val_total, self.train_var[idx][np.newaxis, ...]], axis=0
            )
            val_labels = np.concatenate(
                [val_labels, self.train_labels[idx][np.newaxis, ...]], axis=0
            )
        train_ds = (
            tf.data.Dataset.from_tensor_slices(
                (self.train_total, train_labels.reshape(-1))
            )
            .batch(train_batch_size)
            .prefetch(512)
            .shuffle(3000, seed=3)
        )
        val_ds = tf.data.Dataset.from_tensor_slices(
            (self.val_total, val_labels.reshape(-1))
        ).batch(len(val_labels))
        return train_ds, val_ds

    def get_train_test_ds(self, train_batch_size=64):
        train_ds = (
            tf.data.Dataset.from_tensor_slices((self.train_var, self.train_labels))
            .batch(train_batch_size)
            .prefetch(512)
            .shuffle(3000, seed=777)
        )
        test_ds = tf.data.Dataset.from_tensor_slices(
            (self.test_var, self.test_labels)
        ).batch(self.test_labels.shape[0])
        return train_ds, test_ds
