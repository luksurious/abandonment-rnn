import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
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

pre_train_time = time.time()

UNITS = 50
PATIENCE = 1
FOLD = 5

# Baseline
# ACC: 71%, Prec: 55%, Recall: 71%, F1: 62%, AUC: 50%
# train_model(train_data_km_last[0], train_data_km_last[2],
#             "baseline-no-time", PATIENCE, FOLD, units=UNITS, undersample=False, augment_train=False)

# augment only training data: still very few 0s in test data
# ACC: 0.74
# Precision: 74.02%
# Recall: 74.00%
# F-measure: 72.28%
# AUC: 61.33%
# 4th: less augmentation
# ACC: 0.73
# Precision: 75.36%
# Recall: 73.00%
# F-measure: 73.54%
# AUC: 67.33%
#
train_model(train_data_km_last[0], train_data_km_last[2],
            "no-time-attention", PATIENCE, FOLD, units=UNITS, undersample=True, augment_train=True)
train_model(train_data_km_last[0], train_data_km_last[2],
            "no-time-attention", PATIENCE, 1, units=UNITS, undersample=True, augment_train=True)

# different augmentation inside (80% train)
# ACC: 0.72
# Precision: 70.16%
# Recall: 72.00%
# F-measure: 70.52%
# AUC: 61.33%

# 70% train
# ACC: 0.65
# Precision: 69.49%
# Recall: 64.67%
# F-measure: 66.20%
# AUC: 61.59%

# 90% train
# ACC: 0.64
# Precision: 71.94%
# Recall: 64.00%
# F-measure: 64.46%
# AUC: 64.76%
# train_model(train_data_km_last[0], train_data_km_last[2],
#             "no-time-attention", PATIENCE, FOLD, units=UNITS, undersample=True, augment_train=True)

# augment 50% of 0s before splitting, and training data in full
# ACC: 0.82
# Precision: 82.84%
# Recall: 81.76%
# F-measure: 81.83%
# AUC: 80.61%

# ACC: 0.81
# Precision: 81.54%
# Recall: 80.59%
# F-measure: 80.73%
# AUC: 79.70%

# x, y = train_data_km_last[0], train_data_km_last[2]
# middle_point = int(len(x)/2)
# x_n, y_n = augment_coord(x[:middle_point], y[:middle_point], cutoff=False, varycoord=False, varycutoff=True,
#                      only0=True, varycount=1, cutoff_end=False, cutoff_list=[4])
#
# x = np.concatenate([x, x_n])
# y = np.concatenate([y, y_n])
#
# train_model(x, y,
#             "no-time-attention", PATIENCE, FOLD, units=UNITS, undersample=True, augment_train=True)

# augment full data set, only 0s?
# ACC: 0.73
# Precision: 74.80%
# Recall: 73.10%
# F-measure: 72.51%
# AUC: 72.67%
# x, y = augment_coord(train_data_km_last[0], train_data_km_last[2], cutoff=False, varycoord=False, varycutoff=True,
#                      only0=True)
# train_model(x, y, "no-time-attention", PATIENCE, FOLD, units=UNITS, undersample=True, augment_train=False)

# augment full data set, 0s, 1s separately; more augmentation
# too suspicious
# ACC: 0.93
# Precision: 93.01%
# Recall: 92.93%
# F-measure: 92.92%
# AUC: 92.93%
# x, y = augment_coord(train_data_km_last[0], train_data_km_last[2], cutoff=False, varycoord=False, varycutoff=True,
#                      only1=True, varycount=1, cutoff_list=[5, 10], cutoff_limit=5)
# x, y = augment_coord(x, y, cutoff=False, varycoord=False, varycutoff=True,
#                      only0=True, cutoff_list=[2, 3, 4], cutoff_limit=2.5, varycount=3)
# train_model(x, y, "no-time-attention", PATIENCE, FOLD, units=UNITS, undersample=True, augment_train=False)

# augment full data set: too suspicious
# ACC: 0.96
# Precision: 95.74%
# Recall: 95.60%
# F-measure: 95.59%
# AUC: 95.61%
# x, y = augment_coord(train_data_km_last[0], train_data_km_last[2], cutoff=False, varycoord=False, varycutoff=True,
#                      only0=False)
# train_model(x, y, "no-time-attention", PATIENCE, FOLD, units=UNITS, undersample=True, augment_train=False)


print("Training time: %.2f" % ((time.time()-pre_train_time)/60))
