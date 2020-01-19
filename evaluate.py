import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from keras.engine.saving import load_model
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, save_model
from tensorflow.python.keras.layers import Dense, LSTM, Bidirectional, Dropout, Embedding, TimeDistributed, GRU
from tensorflow.python.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, save_model, load_model
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

import random

from data_provider import *
from mouse_augment import augment_coord
from training import train_model


np.random.seed(123)
random.seed(123)

user_info = read_kge_log()
data_frames = read_csv_logs(user_info)
data_nc_with_km_sr, data_nc_without_km_sr = filter_data_frames(data_frames)

data_nc_with_km_sr_last = get_data_last_only(data_nc_with_km_sr)
data_nc_without_km_sr_last = get_data_last_only(data_nc_without_km_sr)
data_nc_with_km_sr_solo = get_solo_only(data_nc_with_km_sr)
data_nc_without_km_sr_solo = get_solo_only(data_nc_without_km_sr)

users_tasks = map_user_tasks(data_frames, user_info)

data_nc_with_km_sr_last_filtered = filter_by_user_info(user_info, data_nc_with_km_sr_last, users_tasks)
data_nc_with_km_sr_solo_filtered = filter_by_user_info(user_info, data_nc_with_km_sr_solo, users_tasks)


data_filtered_last = data_nc_with_km_sr_last_filtered + data_nc_without_km_sr_last
data_filtered_solo = data_nc_with_km_sr_solo_filtered + data_nc_without_km_sr_solo


VERBOSE = True

train_data_last = extract_data(data_filtered_last, users_tasks, user_info)
train_data_solo = extract_data(data_filtered_solo, users_tasks, user_info)

train_data_km_last = extract_data(data_nc_with_km_sr_last_filtered, users_tasks, user_info)
train_data_km_solo = extract_data(data_nc_with_km_sr_solo_filtered, users_tasks, user_info)

# mouse_moves, mouse_moves_time, au_labels, af_labels, auf_labels = extract_data(data_filtered_last)

model_name = "mouse-aband-no-time-attention"
folds = 5
folded_scores = np.zeros((folds, 5))

x, y = train_data_km_last[0], train_data_km_last[2]

for fold in range(folds):
    model = load_model('models/%s_%d.h5' % (model_name, fold))

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify=y, random_state=123)

    loss, acc, _ = model.evaluate(x_test, y_test)

    folded_scores[fold, 0] = acc

    probs = model.predict(x_test)
    y_pred = np.array([round(x[0]) for x in probs], dtype=np.int)

    # Let's see how good those predictions were.
    precision, recall, fmeasure, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    folded_scores[fold, 1] = precision
    folded_scores[fold, 2] = recall
    folded_scores[fold, 3] = fmeasure

    # Finally compute the ROC AUC to see the discriminative power of the model.
    # binary_labels = [p == y_test[i] for i, p in enumerate(y_pred)]
    auc = roc_auc_score(y_test, y_pred, average='weighted')
    folded_scores[fold, 4] = auc

print('\n\nAverage stats:')
print('ACC: {:.2f}'.format(np.mean(folded_scores[:, 0])))
print('Precision: {:.2f}%'.format(np.mean(folded_scores[:, 1]) * 100))
print('Recall: {:.2f}%'.format(np.mean(folded_scores[:, 2]) * 100))
print('F-measure: {:.2f}%'.format(np.mean(folded_scores[:, 3]) * 100))
print('AUC: {:.2f}%'.format(np.mean(folded_scores[:, 4]) * 100))

