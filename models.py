from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Embedding, TimeDistributed, GRU, \
    BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam
from tensorflow.keras import metrics
# import tensorflow_addons as tfa
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention


def create_model_template(layers: int, units, shape, use_attention_first=False, use_attention_middle=False, lr=3e-4,
                          optimizer='Adam', dropout=0.2, dropout_last_only=False):
    model_metrics = [
        metrics.BinaryAccuracy(name='acc'),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc')
    ]

    model = Sequential()

    if not isinstance(units, list):
        units = [units] * layers
    elif len(units) < layers:
        units = [units[0]] * layers

    model.add(Bidirectional(LSTM(units[0], return_sequences=layers > 1 or use_attention_first), input_shape=shape))
    # model.add(Bidirectional(tfa.rnn.cell.LayerNormLSTMCell(units[0], return_sequences=layers > 1), input_shape=shape))

    if use_attention_first:
        if layers > 1:
            model.add(SeqSelfAttention())
        else:
            model.add(SeqWeightedAttention())

    for i in range(1, layers):
        if use_attention_middle:
            model.add(SeqSelfAttention())

        if dropout_last_only is False:
            model.add(Dropout(dropout))

        model.add(Bidirectional(LSTM(units[i], return_sequences=layers > i + 1 or use_attention_middle)))

    if use_attention_middle:
        model.add(SeqWeightedAttention())

    model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    if optimizer == 'SGD':
        optimizer = SGD(lr=lr)
    if optimizer == 'RMSprop':
        optimizer = RMSprop(lr=lr)
    if optimizer == 'Adadelta':
        optimizer = Adadelta(lr=lr)
    if optimizer == 'Adagrad':
        optimizer = Adagrad(lr=lr)
    if optimizer == 'Nadam':
        optimizer = Nadam(lr=lr)
    if optimizer == 'Adamax':
        optimizer = Adamax(lr=lr)
    else:
        optimizer = Adam(lr=lr)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=model_metrics)

    return model


def create_model(units, shape):
    return create_model_template(3, units, shape)
