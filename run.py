import time
import warnings
import numpy as np
import random
import argparse
warnings.simplefilter(action='ignore', category=FutureWarning)


def setup_arguments():
    parser = argparse.ArgumentParser(description='Run an DL experiment on mouse movement data for abandonment research')

    parser.add_argument('--layers', type=int, default=3, help='Depth of RNN')
    parser.add_argument('--units', type=int, nargs='?', default=50, help='Units in RNN per layer')
    parser.add_argument('--patience', type=int, default=10, help='Patience of keras callback')
    parser.add_argument('--stop_val', type=str, default='val_f1_score', choices=['val_f1_score', 'val_auc', 'val_acc'],
                        help='Which metric to use for early stopping')  # TODO
    parser.add_argument('--folds', type=int, default=5, help='KFolds for validation')
    parser.add_argument('--use_time', action='store_true', help='Use action delta as input')
    parser.add_argument('--label', type=str, choices=['au', 'af', 'auf'], default='au', help='Type of label to use')
    parser.add_argument('--only_solo', action='store_true', help='Ignore sessions with multiple searches')

    parser.add_argument('--normalize_coords', action='store_true', help='normalize coordinates')  # TODO

    parser.add_argument('--no_augment', action='store_true', help='Disable augmentation')
    # TODO different augmentation
    # TODO self-attention

    parser.add_argument('--no_undersample', action='store_true', help='Disable undersampling')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--dropout_only_last', action='store_true', help='Only apply dropout at the last layer')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'SDG', 'RMSprop', 'AdaDelta', 'AdaGrad'],
                        default='Adam', help='Learning rate')  # TODO

    # Execution arguments
    parser.add_argument('-v', '--verbose', action="store_true", help="Print everything")
    parser.add_argument('--seed', type=int, default=123, help="Base seed for the different simulation runs")
    parser.add_argument('--file_desc', type=str, default="", help="name of files for this run")
    parser.add_argument('mode', type=str, default="model", choices=["model", "evaluate"],
                        help="Either find the best `model` based on validation data, or `evaluate` given model on "
                             "test data")  # TODO

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
    # train_model(train_data_km_last[0], train_data_km_last[2],
    #             "baseline-no-time", PATIENCE, FOLD, units=UNITS, undersample=False, augment_train=False)

    if args.only_solo:
        data_nc_with_km_sr_solo = get_solo_only(data_nc_with_km_sr)
        data_nc_with_km_sr_solo_filtered = filter_by_user_info(user_info, data_nc_with_km_sr_solo, users_tasks)
        train_data_km_solo = extract_data(data_nc_with_km_sr_solo_filtered, users_tasks, user_info)
        data = train_data_km_solo
    else:
        data_nc_with_km_sr_last = get_data_last_only(data_nc_with_km_sr)
        data_nc_with_km_sr_last_filtered = filter_by_user_info(user_info, data_nc_with_km_sr_last, users_tasks)
        train_data_km_last = extract_data(data_nc_with_km_sr_last_filtered, users_tasks, user_info)
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

    if args.mode == 'model':
        train_model(args, x, y, args.file_desc, args.patience, args.fold, units=args.units,
                    undersample=args.no_undersample is False, augment_train=args.no_augment is False)
    else:
        print("TODO")

    print("Training time: %.2f" % ((time.time() - pre_train_time) / 60))
