import logging
import os
import random

from fbeta import FBetaMetricCallback

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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
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
        print("\n---------------------------------\n")
        print("Got %d data points. 0: %d, 1: %d" % (len(x), len(y) - np.count_nonzero(y), np.count_nonzero(y)))

    if is_eval is False or folds == 1:
        # Separate test set
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            train_size=args.train_split, stratify=y,
                                                            random_state=args.seed)

        if optimizing is False:
            print("Test: %d data points, 0: %d, 1: %d" % (len(x_test), len(y_test) - np.count_nonzero(y_test),
                                                          np.count_nonzero(y_test)))
    else:
        x_train, y_train = x, y

    folded_scores = []
    histories = []

    if folds > 1:
        kf = StratifiedKFold(n_splits=folds)

        for index, (train_index, val_index) in enumerate(kf.split(x_train, y_train)):
            if optimizing is False:
                if args.verbose:
                    print("\nSplit %d..." % index)
                else:
                    print("Split %d..." % index)

            if is_eval is False:
                x_train_fold, x_val = x_train[train_index], x_train[val_index]
                y_train_fold, y_val = y_train[train_index], y_train[val_index]
            else:
                x_train_fold, x_test = x_train[train_index], x_train[val_index]
                y_train_fold, y_test = y_train[train_index], y_train[val_index]

                x_train_fold, x_val, y_train_fold, y_val = train_test_split(x_train_fold, y_train_fold,
                                                                            train_size=args.train_split,
                                                                            stratify=y_train_fold,
                                                                            random_state=args.seed)

            if optimizing is False and args.verbose:
                print("Train: %d data points, 0: %d, 1: %d" % (len(x_train_fold),
                                                               len(y_train_fold) - np.count_nonzero(y_train_fold),
                                                               np.count_nonzero(y_train_fold)))
                print("Val: %d data points, 0: %d, 1: %d" % (len(x_val), len(y_val) - np.count_nonzero(y_val),
                                                             np.count_nonzero(y_val)))

            x_train_fold, y_train_fold = postsplit_resampling(args, augment_train, undersample, x_test, x_train_fold,
                                                              y_train_fold)

            if optimizing is False and args.verbose:
                print("After resampling: %d data points, 0: %d, 1: %d"
                      % (len(x_train_fold), len(y_train_fold) - np.count_nonzero(y_train_fold),
                         np.count_nonzero(y_train_fold)))

            # in model parameter mode: use splitting as validation
            # in model evaluation mode: use splitting as test set
            if is_eval is False:
                x_test = x_val
                y_test = y_val

            iterator = range(1)
            if repeated is True:
                iterator = range(10)

            for _ in iterator:
                history = run_train(args, x_train_fold, y_train_fold, x_val, y_val, x_test, y_test, folded_scores,
                                    index, prefix, patience, units)
                histories.append(history)
    else:

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

        history = run_train(args, x_train_fold, y_train_fold, x_val, y_val, x_test, y_test,
                            folded_scores, 0, prefix, patience, units)
        histories.append(history)

    folded_scores = np.array(folded_scores)
    if optimizing is False:
        print('\n\nAverage stats:')
        print(tt.to_string([
            ["Mean"] + ["%.2f" % (item*100)
                        for item in [np.mean(folded_scores[:, i]) for i in range(5)]],

            ["95% CI"] + ["+- %.2f" % (item*100)
                          for item in [np.mean(folded_scores[:, i]) -
                                       st.t.interval(0.95, len(folded_scores[:, i]) - 1,
                                                     loc=np.mean(folded_scores[:, i]),
                                                     scale=st.sem(folded_scores[:, i]))[0] for i in range(5)]],
        ], ["", "Loss", "Precision", "Recall", "F1-Score", "AUC ROC"], alignment="lrrrrr"))

        print(folded_scores)

        plot_history(histories, prefix)

    return folded_scores


def postsplit_resampling(args, augment_train, undersample, x_test, x_train, y_train):
    if augment_train:
        x_train, y_train = augment_coord(x_train, y_train, cutoff=True, varycoord=True, varycutoff=False,
                                         cutoff_list=[2, 4], varycount=3, cutoff_end=True, cutoff_limit=5
                                         )

    if args.oversample != '':
        if args.oversample == 'smote':
            ros = SMOTE(random_state=args.seed)
        elif args.oversample == 'adasyn':
            ros = ADASYN(random_state=args.seed)
        else:
            ros = RandomOverSampler(random_state=args.seed)
        x_train, y_train = ros.fit_resample(x_train.reshape(x_train.shape[0], -1), y_train)
        x_train = x_train.reshape(x_train.shape[0], x_test.shape[1], -1)
    elif undersample:
        ros = RandomUnderSampler(random_state=args.seed)
        x_train, y_train = ros.fit_resample(x_train.reshape(x_train.shape[0], -1), y_train)
        x_train = x_train.reshape(x_train.shape[0], x_test.shape[1], -1)

    return x_train, y_train


def run_train(args, x_train, y_train, x_val, y_val, x_test, y_test, folded_scores, index, prefix, patience, units):
    # Set callbacks, for monitoring progress.
    # cb_tensorboard = TensorBoard(log_dir='mouse_logs/%s_%d' % (prefix, index))
    cb_earlystopping = EarlyStopping(patience=patience, restore_best_weights=True,
                                     monitor=args.stop_val, mode='max')
    # cb_checkpoint = ModelCheckpoint('mouse_logs/best.h5', save_best_only=True)

    cb_fbeta = FBetaMetricCallback((x_val, y_val), threshold=args.threshold)

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train),
                                                      y_train)

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

    # Evaluate the model.
    loss, acc, _, _, _ = model.evaluate(x_test, y_test, verbose=args.verbose)

    # model = load_model('mouse_logs/best.h5')

    # loss, acc, _ = model.evaluate(x_test, y_test)

    # folded_scores[index, 0] = acc

    probs = model.predict(tf.cast(x_test, dtype=tf.float32))
    y_pred = np.array([int(x[0] > args.threshold) for x in probs], dtype=np.int)

    # Let's see how good those predictions were.
    precision, recall, fmeasure, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    # folded_scores[index, 1] = precision
    # folded_scores[index, 2] = recall
    # folded_scores[index, 3] = fmeasure

    # Finally compute the ROC AUC to see the discriminative power of the model.
    # binary_labels = [p == y_test[i] for i, p in enumerate(y_pred)]
    auc = roc_auc_score(y_test, y_pred, average='weighted')
    # folded_scores[index, 4] = auc

    folded_scores.append([
        loss,
        precision,
        recall,
        fmeasure,
        auc
    ])

    # Save the model.
    # save_model(model, '{}.h5'.format("models/mouse-aband-%s_%d" % (prefix, index)))
    # print(y_pred)

    return history


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
