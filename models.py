from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Embedding, TimeDistributed, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics


def create_model_template(layers: int, units, shape):
    model_metrics = [
        metrics.BinaryAccuracy(name='acc'),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc')
    ]

    model = Sequential()

    if not isinstance(units, list):
        units = [units]*layers

    model.add(Bidirectional(LSTM(units[0], return_sequences=layers > 1), input_shape=shape))

    for i in range(1, layers):
        # model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(units[i], return_sequences=layers > i+1)))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=model_metrics)

    return model


def create_model(units, shape):
    return create_model_template(3, units, shape)
