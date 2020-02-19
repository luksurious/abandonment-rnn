import logging
import os
import random

from fbeta import FBetaMetricCallback
from mcc import MCCMetricCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import termtables as tt
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, precision_score, recall_score, \
    f1_score
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import tensorflow.keras as keras

from models import create_model, create_model_template
from mouse_augment import augment_coord


# import tensorflow.compat.v1 as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.Session(config=config)


def train_model(args, x, y, prefix, patience=3, folds=10, undersample=False, units=50, augment_train=True,
                is_eval=False, repeated=False, optimizing=False):
    np.random.seed(args.seed)
    random.seed(args.seed)
    if repeated is False:
        tf.random.set_seed(args.seed)

    if optimizing is False:
        print("---------------------------------")
        print("Got %d data points. 0: %d, 1: %d" % (len(x), len(y) - np.count_nonzero(y), np.count_nonzero(y)))

    folded_scores = []
    folded_val_scores = []
    histories = []

    if folds > 1:
        kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed)

        for index, (train_index, test_index) in enumerate(kf.split(x, y)):
            exec_folded_training(args, augment_train, folded_scores, folded_val_scores,
                                 histories, index, is_eval, optimizing, patience,
                                 prefix, repeated, test_index, train_index, undersample, units, x, y)
    else:
        exec_single_training(args, augment_train, folded_scores, histories, patience, prefix, undersample, units, x, y)

    folded_scores = np.array(folded_scores)
    folded_val_scores = np.array(folded_val_scores)
    if optimizing is False:

        print('\n\nValidation data stats:')
        print(tt.to_string([
            ["Mean"] + ["%.2f" % (item * 100)
                        for item in [np.mean(folded_val_scores[:, i]) for i in range(5)]],

            ["95% CI"] + ["+- %.2f" % (item * 100)
                          for item in [np.mean(folded_val_scores[:, i]) -
                                       st.t.interval(0.95, len(folded_val_scores[:, i]) - 1,
                                                     loc=np.mean(folded_val_scores[:, i]),
                                                     scale=st.sem(folded_val_scores[:, i]))[0] for i in range(5)]],

            ["SD"] + ["+- %.2f" % (item * 100)
                      for item in [np.std(folded_val_scores[:, i]) for i in range(5)]],
        ], ["", "Loss", "Precision", "Recall", "F1-Score", "AUC ROC"], alignment="lrrrrr"))

        print('\n\nTest data stats:')
        print(tt.to_string([
            ["Mean"] + ["%.2f" % (item * 100)
                        for item in [np.mean(folded_scores[:, i]) for i in range(5)]],

            ["95% CI"] + ["+- %.2f" % (item * 100)
                          for item in [np.mean(folded_scores[:, i]) -
                                       st.t.interval(0.95, len(folded_scores[:, i]) - 1,
                                                     loc=np.mean(folded_scores[:, i]),
                                                     scale=st.sem(folded_scores[:, i]))[0] for i in range(5)]],

            ["SD"] + ["+- %.2f" % (item * 100)
                      for item in [np.std(folded_scores[:, i]) for i in range(5)]],
        ], ["", "Loss", "Precision", "Recall", "F1-Score", "AUC ROC"], alignment="lrrrrr"))

        print(folded_scores)

        if args.verbose:
            plot_history(histories, prefix)

    return folded_scores, folded_val_scores


def exec_folded_training(args, augment_train, folded_scores, folded_val_scores, histories, index, is_eval, optimizing,
                         patience, prefix, repeated, test_index, train_index, undersample, units, x, y):
    # in each fold, first separate the training and test set
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    if optimizing is False:
        print("Split %d..." % index)

    kf = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    if args.no_nested or args.dummy:
        iterator = [next(kf.split(x_train, y_train))]
    else:
        iterator = kf.split(x_train, y_train)

    for train_index, val_index in iterator:
        # then separate the validation set from the training set
        x_train_fold, x_val = x_train[train_index], x_train[val_index]
        y_train_fold, y_val = y_train[train_index], y_train[val_index]

        x_train_fold, y_train_fold = postsplit_resampling(args, augment_train, undersample, x_test, x_train_fold,
                                                          y_train_fold)

        print_fold_info(args, index, optimizing, x_test, x_train_fold, x_val, y_test, y_train_fold, y_val)

        # if optimizing is False and args.verbose:
        #     print("After resampling: %d items, 0: %d, 1: %d"
        #           % (len(x_train_fold), len(y_train_fold) - np.count_nonzero(y_train_fold),
        #              np.count_nonzero(y_train_fold)))

        history = run_train(args, x_train_fold, y_train_fold, x_val, y_val, x_test, y_test, folded_scores,
                            folded_val_scores, index, prefix, patience, units)
        histories.append(history)


def print_fold_info(args, index, optimizing, x_test, x_train_fold, x_val, y_test, y_train_fold, y_val):
    # print some information
    if optimizing is False and args.verbose:
        print("Split %d..." % index)
        print(tt.to_string([
            [
                "%d items, 0: %d, 1: %d" % (len(x_train_fold),
                                            len(y_train_fold) - np.count_nonzero(y_train_fold),
                                            np.count_nonzero(y_train_fold)),
                "%d data points, 0: %d, 1: %d" % (len(x_val), len(y_val) - np.count_nonzero(y_val),
                                                  np.count_nonzero(y_val)),
                "%d items, 0: %d, 1: %d" % (len(x_test), len(y_test) - np.count_nonzero(y_test),
                                            np.count_nonzero(y_test))
            ]
        ], ["Train", "Validation", "Test"], alignment="lll"))


def exec_single_training(args, augment_train, folded_scores, histories, patience, prefix, undersample, units, x, y):
    # Separate test set
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=args.train_split, stratify=y,
                                                        random_state=args.seed)
    x_train_fold, x_val, y_train_fold, y_val = train_test_split(x_train, y_train, train_size=0.8,
                                                                stratify=y_train, random_state=args.seed)
    print("Train: %d data points, 0: %d, 1: %d" % (len(x_train_fold),
                                                   len(y_train_fold) - np.count_nonzero(y_train_fold),
                                                   np.count_nonzero(y_train_fold)))
    print("Val: %d data points, 0: %d, 1: %d" % (len(x_val), len(y_val) - np.count_nonzero(y_val),
                                                 np.count_nonzero(y_val)))
    x_train_fold, y_train_fold = postsplit_resampling(args, augment_train, undersample, x_test, x_train_fold,
                                                      y_train_fold)
    print("After resampling: %d data points, 0: %d, 1: %d"
          % (len(x_train_fold), len(y_train_fold) - np.count_nonzero(y_train_fold),
             np.count_nonzero(y_train_fold)))
    history = run_train(args, x_train_fold, y_train_fold, x_val, y_val, x_test, y_test, folded_scores, [],
                        0, prefix, patience, units)
    histories.append(history)


def postsplit_resampling(args, augment_train, undersample, x_test, x_train, y_train):
    if augment_train:
        x_train, y_train = custom_augment(args, x_train, y_train)

    if args.oversample != '':
        if args.oversample == 'smote' or args.oversample == 'smote+':
            ros = SMOTE(random_state=args.seed)
        elif args.oversample == 'adasyn' or args.oversample == 'adasyn+':
            ros = ADASYN(random_state=args.seed)
        else:
            ros = RandomOverSampler(random_state=args.seed)

        x_train, y_train = ros.fit_resample(x_train.reshape(x_train.shape[0], -1), y_train)
        x_train = x_train.reshape(x_train.shape[0], x_test.shape[1], -1)

        if args.oversample == 'adasyn+' or args.oversample == 'smote+':
            x_train, y_train = custom_augment(args, x_train, y_train)

    elif undersample:
        ros = RandomUnderSampler(random_state=args.seed)
        x_train, y_train = ros.fit_resample(x_train.reshape(x_train.shape[0], -1), y_train)
        x_train = x_train.reshape(x_train.shape[0], x_test.shape[1], -1)

    return x_train, y_train


def custom_augment(args, x_train, y_train):
    if args.aug_mode == 'vary+cut':
        cutoff = True
        vary = True
        varycut = False
    elif args.aug_mode == 'vary':
        cutoff = False
        vary = True
        varycut = False
    elif args.aug_mode == 'cut':
        cutoff = True
        vary = False
        varycut = False
    else:
        cutoff = False
        vary = False
        varycut = True
    x_train, y_train = augment_coord(x_train, y_train, cutoff=cutoff, varycoord=vary, varycutoff=varycut,
                                     cutoff_list=args.aug_cutoff_lens, varycount=args.aug_varycount,
                                     cutoff_end=args.aug_cutoff_end, cutoff_limit=args.aug_cutoff_limit,
                                     balance=args.aug_no_balanced is False, offset_dupes=args.aug_offset,
                                     var_strength=args.aug_var_strength
                                     )
    return x_train, y_train


def run_train(args, x_train, y_train, x_val, y_val, x_test, y_test, folded_scores, folded_val_scores, index, prefix,
              patience, units):
    # Set callbacks, for monitoring progress.
    # cb_tensorboard = TensorBoard(log_dir='mouse_logs/%s_%d' % (prefix, index))
    mode = 'min' if args.stop_val == 'val_loss' else 'max'
    cb_earlystopping = EarlyStopping(patience=patience, restore_best_weights=True,
                                     monitor=args.stop_val, mode=mode)
    # cb_checkpoint = ModelCheckpoint('mouse_logs/best.h5', save_best_only=True)

    cb_fbeta = FBetaMetricCallback((x_val, y_val), threshold=args.threshold)
    # cb_mcc = MCCMetricCallback((x_val, y_val), threshold=args.threshold)

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train),
                                                      y_train)

    class_weights /= np.min(class_weights)

    if args.dummy:
        majority_class = 0  # np.argmin(class_weights)
        inp = keras.layers.Input(x_train[0].shape)
        out = keras.layers.Lambda(lambda x: [tf.ones((1,), dtype=tf.float32) * majority_class])(inp)
        model = keras.models.Model(inp, out)

        # train_predictions = model.predict(x_train, batch_size=args.batch_size)
        # print(train_predictions)
        model.compile(loss='binary_crossentropy', metrics=['accuracy'])
        history = []
    else:
        # Train the model.
        model = create_model_template(args.layers, units, x_train[0].shape, args.attention_first, args.attention_middle,
                                      args.lr, args.optimizer, args.dropout, args.dropout_only_last)
        # print(model.summary())
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=100,
            class_weight=class_weights,
            callbacks=[cb_fbeta, cb_earlystopping],
            verbose=args.verbose,
            batch_size=args.batch_size
        )

    # train_predictions = model.predict(x_train, batch_size=args.batch_size)

    # plt.clf()
    # plot_roc("Train", y_train, train_predictions)

    # Evaluate the model.
    loss, precision, recall, fmeasure, auc, y_pred = evaluate_model(args, model, x_test, y_test)
    folded_scores.append([loss, precision, recall, fmeasure, auc])
    # plot_roc("Test", y_test, y_pred)

    loss, precision, recall, fmeasure, auc, y_pred = evaluate_model(args, model, x_val, y_val)
    folded_val_scores.append([loss, precision, recall, fmeasure, auc])
    # plot_roc("Val", y_val, y_pred)
    # plt.show()

    # Save the model.
    # save_model(model, '{}.h5'.format("models/mouse-aband-%s_%d" % (prefix, index)))
    # print(y_pred)

    return history


def evaluate_model(args, model, x_test, y_test):
    if args.dummy:
        loss, acc = model.evaluate(x_test, y_test, verbose=args.verbose, batch_size=args.batch_size)
        probs = model.predict(tf.cast(x_test, dtype=tf.float32), batch_size=args.batch_size)
        y_pred = np.array([int(x > args.threshold) for x in probs], dtype=np.int)
    else:
        loss, acc, _, _, _ = model.evaluate(x_test, y_test, verbose=args.verbose, batch_size=args.batch_size)
        probs = model.predict(tf.cast(x_test, dtype=tf.float32), batch_size=args.batch_size)
        y_pred = np.array([int(x[0] > args.threshold) for x in probs], dtype=np.int)

    precision, recall, fmeasure, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    auc = roc_auc_score(y_test, y_pred)

    return loss, precision, recall, fmeasure, auc, y_pred


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, thresholds = roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.legend()
    # plt.xlim([-0.5,20])
    # plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def plot_history(histories, suffix):
    for i, h in enumerate(histories):
        plt.plot(h.history['acc'], color='blue')
        plt.plot(h.history['val_acc'], color='orange')

    plt.title('model accuracy fold')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('models/model-accuracy_%s.png' % (suffix))
    plt.show()
    plt.clf()

    for i, h in enumerate(histories):
        plt.plot(h.history['loss'], color='blue')
        plt.plot(h.history['val_loss'], color='orange')

    plt.title('model loss fold')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('models/model-loss_%s.png' % (suffix))
    plt.show()
    plt.clf()
