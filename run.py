import time
import warnings
import numpy as np
import random
import argparse

import optuna

warnings.simplefilter(action='ignore', category=FutureWarning)


def setup_arguments():
    parser = argparse.ArgumentParser(description='Run an DL experiment on mouse movement data for abandonment research')

    parser.add_argument('--layers', type=int, default=3, help='Depth of RNN')
    parser.add_argument('--units', type=int, nargs='*', default=50, help='Units in RNN per layer')
    parser.add_argument('--patience', type=int, default=10, help='Patience of keras callback')
    parser.add_argument('--stop_val', type=str, default='val_f1_score', choices=['val_f1_score', 'val_auc', 'val_loss'],
                        help='Which metric to use for early stopping')
    parser.add_argument('--folds', type=int, default=5, help='KFolds for validation')
    parser.add_argument('--use_time', action='store_true', help='Use action delta as input')
    parser.add_argument('--label', type=str, choices=['au', 'af', 'auf'], default='au', help='Type of label to use')
    parser.add_argument('--only_solo', action='store_true', help='Ignore sessions with multiple searches')
    parser.add_argument('--max_events', type=int, default=50, help='Max number of last mouse movements to consider')
    parser.add_argument('--min_events', type=int, default=5, help='Min number of mouse movements to consider')

    parser.add_argument('--standardize', action='store_true', help='Disable augmentation')
    parser.add_argument('--normalize', action='store_true', help='normalize coordinates')
    parser.add_argument('--norm_time', action='store_true', help='normalize coordinates')
    parser.add_argument('--reset_origin', action='store_true', help='normalize coordinates')

    parser.add_argument('--no_augment', action='store_true', help='Disable augmentation')
    parser.add_argument('--aug_varycount', type=int, default=3, help='Disable augmentation')
    parser.add_argument('--aug_var_strength', type=int, default=2, help='Disable augmentation')
    parser.add_argument('--aug_cutoff_lens', type=int, nargs='*', default=[2, 3, 4], help='Disable augmentation')
    parser.add_argument('--aug_cutoff_limit', type=int, default=5, help='Disable augmentation')
    parser.add_argument('--aug_cutoff_end', action='store_true', help='Disable augmentation')
    parser.add_argument('--aug_offset', action='store_true', help='Disable augmentation')

    parser.add_argument('--attention_first', action='store_true', help='Disable augmentation')
    parser.add_argument('--attention_middle', action='store_true', help='Disable augmentation')

    parser.add_argument('--no_undersample', action='store_true', help='Disable undersampling')
    parser.add_argument('--oversample', type=str, default='', choices=['', 'random', 'smote', 'adasyn'],
                        help='Use automatic oversampling instead of undersampling or augmentation')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--dropout_only_last', action='store_true', help='Only apply dropout at the last layer')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classifier threshold')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--optimizer', type=str,
                        choices=['Adam', 'SDG', 'RMSprop', 'Adadelta', 'Adagrad', 'Nadam', 'Adamax'],
                        default='Adam', help='Learning rate')

    # Execution arguments
    parser.add_argument('-v', '--verbose', action="store_true", help="Print everything")
    parser.add_argument('--optuna', action="store_true", help="Use optuna to optimize hyperparams")
    parser.add_argument('--opt_trials', type=int, default=50, help="Number of optuna trials")
    parser.add_argument('--seed', type=int, default=123, help="Base seed for the different simulation runs")
    parser.add_argument('--file_desc', type=str, default="", help="name of files for this run")
    parser.add_argument('--repeated', action='store_true', help='perform repeated evaluation to average '
                                                                'optimizer randomness')
    parser.add_argument('--train_split', type=float, default=0.8, help='Split of train/test|val')
    parser.add_argument('mode', type=str, default="model", choices=["model", "evaluate"],
                        help="Either find the best `model` based on validation data, or `evaluate` given model on "
                             "test data")

    return parser


if __name__ == '__main__':
    parser = setup_arguments()
    args = parser.parse_args()
    # model_info = describe_arguments(args)

    from data_provider import read_csv_logs, read_kge_log, filter_data_frames, filter_by_user_info, get_solo_only, \
        get_data_last_only, map_user_tasks, extract_data
    from training import train_model

    np.random.seed(args.seed)
    random.seed(args.seed)

    user_info = read_kge_log()
    data_frames = read_csv_logs(user_info)
    data_nc_with_km_sr, data_nc_without_km_sr = filter_data_frames(data_frames)

    users_tasks = map_user_tasks(data_frames, user_info)

    pre_train_time = time.time()

    # Baseline
    # ACC: 71%, Prec: 55%, Recall: 71%, F1: 62%, AUC: 50%

    if args.only_solo:
        data_nc_with_km_sr_solo = get_solo_only(data_nc_with_km_sr)
        data_nc_with_km_sr_solo_filtered = filter_by_user_info(user_info, data_nc_with_km_sr_solo, users_tasks)
        train_data_km_solo = extract_data(data_nc_with_km_sr_solo_filtered, users_tasks, user_info,
                                          args.max_events, args.min_events, args.standardize, args.normalize,
                                          args.reset_origin, args.norm_time)
        data = train_data_km_solo
    else:
        data_nc_with_km_sr_last = get_data_last_only(data_nc_with_km_sr)
        data_nc_with_km_sr_last_filtered = filter_by_user_info(user_info, data_nc_with_km_sr_last, users_tasks)
        train_data_km_last = extract_data(data_nc_with_km_sr_last_filtered, users_tasks, user_info,
                                          args.max_events, args.min_events, args.standardize, args.normalize,
                                          args.reset_origin, args.norm_time)
        data = train_data_km_last

    if args.use_time:
        x = data[1]
    else:
        x = data[0]

    if args.label == 'af':
        y = data[3]
    elif args.label == 'auf':
        y = data[4]
    else:
        y = data[2]

    print("Data points x: %d" % len(x))

    if args.optuna is True:

        def optuna_objective(trial):
            # args.lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
            # args.optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SDG', 'RMSprop', 'Nadam'])
            args.batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])

            args.layers = trial.suggest_categorical('layers', [1, 2, 3])
            args.units = trial.suggest_categorical('units', [25, 50, 100])
            args.dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.5])
            # args.dropout_only_last = trial.suggest_categorical('dropout_only_last', [True, False])
            args.attention_first = trial.suggest_categorical('attention_first', [True, False])
            # args.attention_middle = trial.suggest_categorical('attention_middle', [True, False])

            test_results, val_results = train_model(args, x, y, args.file_desc, args.patience, args.folds,
                                                    units=args.units,
                                                    undersample=args.no_undersample is False,
                                                    augment_train=args.no_augment is False,
                                                    optimizing=True)

            return np.mean(val_results[:, 3])

        study = optuna.create_study(direction='maximize', study_name='mouse_model_opt',
                                    storage='sqlite:///mouse_model_opt.db', load_if_exists=True)
        study.optimize(optuna_objective, n_trials=args.opt_trials)

        print(study.best_params)

    else:
        if args.mode == 'model':
            train_model(args, x, y, args.file_desc, args.patience, args.folds, units=args.units,
                        undersample=args.no_undersample is False, augment_train=args.no_augment is False)
        else:
            train_model(args, x, y, args.file_desc, args.patience, args.folds, units=args.units,
                        undersample=args.no_undersample is False, augment_train=args.no_augment is False, is_eval=True,
                        repeated=args.repeated)

    print("Training time: %.2f" % ((time.time() - pre_train_time) / 60))
