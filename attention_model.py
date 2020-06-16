from tensorflow.keras.layers import (
    LayerNormalization,
    LSTM,
    Layer,
    Bidirectional,
    Dense,
    Dropout,
)
import tensorflow as tf


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
        self.layerNorm = LayerNormalization(axis=-2, input_shape=(-1, 10, 20))
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
        self.lstm = Bidirectional(LSTM(256, input_shape=(-1, 10, 50)))
        self.b_model = XrsInputModel()
        self.sharp_model = SharpInputModel(128)
        self.dropout = Dropout(0.3)
        self.w1 = Dense(128)
        self.w2 = Dense(128)
        self.v = Dense(1)
        self.attention_weights = None
        self.dense = Dense(3, activation="softmax")

    def call(self, inputs, training=False):
        # inputs shape = (batch, 10, 50)
        query = self.lstm(inputs)[:, tf.newaxis, :]  # batch,1,512

        b_input, sharp = tf.split(inputs, [20, 30], axis=-1)
        b_output = self.b_model(b_input)  # batch,512
        sharp_output, sharp_attn_weights = self.sharp_model(sharp)  # batch,512

        b_output = b_output[:, tf.newaxis, :]
        sharp_output = sharp_output[:, tf.newaxis, :]

        values = tf.concat([b_output, sharp_output], axis=1)  # batch,2,512
        score = tf.nn.tanh(self.w1(query) + self.w2(values))  # batch,2,128
        score = self.v(score)  # batch,3,1
        self.attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = self.attention_weights * values  # batch,2,512
        context_vector = tf.reduce_sum(context_vector, axis=1)
        outputs = self.dropout(context_vector, training=training)
        outputs = self.dense(outputs)
        return outputs, sharp_attn_weights, self.attention_weights
