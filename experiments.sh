#!/usr/bin/env bash

DATA_CONFIG="--max_events 50 --min_events 2"
COMMON_ARGS="${DATA_CONFIG} --patience 5 --stop_val val_f1_score --folds 10 --lr 1e-4 --layers 2 --units 100 --dropout 0.3 --batch_size 4"

FOLDER="experiments/"

# Non DL baselines
python -u run.py ${DATA_CONFIG} --all_bad | tee "${FOLDER}baseline-dummy-minority.log"  # minority prediction
python -u run.py ${DATA_CONFIG} --all_good | tee "${FOLDER}baseline-dummy-majority.log"  # majority prediction
python -u run.py ${DATA_CONFIG} --use_classic RF --oversample adasyn | tee "${FOLDER}baseline-classic-RF-adasyn.log"  # Random Forest
python -u run.py ${DATA_CONFIG} --use_classic XGB --oversample adasyn | tee "${FOLDER}baseline-classic-XGB-adasyn.log"  # XGB

# Raw baselines
THIS_ARGS="--no_augment"
python -u run.py $COMMON_ARGS $THIS_ARGS --no_undersample  | tee "${FOLDER}baseline-imbalanced_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --no_undersample --use_time | tee "${FOLDER}baseline-imbalanced_time_f1-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "${FOLDER}baseline-undersample_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "${FOLDER}baseline-undersample_time_f1-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --oversample random  | tee "${FOLDER}baseline-oversample_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample random --use_time | tee "${FOLDER}baseline-oversample_time_f1-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote  | tee "${FOLDER}baseline-smote_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote --use_time | tee "${FOLDER}baseline-smote_time_f1-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --oversample adasyn  | tee "${FOLDER}baseline-adasyn_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample adasyn --use_time | tee "${FOLDER}baseline-adasyn_time_f1-2x100.log"


# Standardized baselines
THIS_ARGS="--no_augment --standardize"
python -u run.py $COMMON_ARGS $THIS_ARGS --no_undersample  | tee "${FOLDER}baseline-stand-imbalanced_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --no_undersample --use_time | tee "${FOLDER}baseline-stand-imbalanced_time_f1-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "${FOLDER}baseline-stand-undersample_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "${FOLDER}baseline-stand-undersample_time_f1-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --oversample random  | tee "${FOLDER}baseline-stand-oversample_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample random --use_time | tee "${FOLDER}baseline-stand-oversample_time_f1-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote  | tee "${FOLDER}baseline-stand-smote_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote --use_time | tee "${FOLDER}baseline-stand-smote_time_f1-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --oversample adasyn  | tee "${FOLDER}baseline-stand-adasyn_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample adasyn --use_time | tee "${FOLDER}baseline-stand-adasyn_time_f1-2x100.log"


# Custom runs
THIS_ARGS=""
python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "${FOLDER}custom_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "${FOLDER}custom_time_f1-2x100.log"

THIS_ARGS="--standardize"
python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "${FOLDER}custom-stand_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "${FOLDER}custom-stand_time_f1-2x100.log"

THIS_ARGS="--aug_no_balanced"
python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "custom-no-balance_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "custom-no-balance_time_f1-2x100.log"

THIS_ARGS="--aug_no_balanced --standardize"
python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "custom-no-balance-stand_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "custom-no-balance-stand_time_f1-2x100.log"


THIS_ARGS="--aug_mode varycutcomb"
python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "custom-augcomb_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "custom-augcomb_time_f1-2x100.log"


THIS_ARGS="--aug_mode vary"
python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "custom-varyonly_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "custom-varyonly_time_f1-2x100.log"

THIS_ARGS="--aug_mode cut"
python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "custom-cutonly_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "custom-cutonly_time_f1-2x100.log"


# Use speed
THIS_ARGS="--no_augment --standardize --use_speed"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote --no_coords | tee "${FOLDER}baseline-speed-nocoords-smote_time_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote --no_coords --use_distances | tee "${FOLDER}baseline-speed+distances-nocoords-smote_time_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote | tee "${FOLDER}baseline-speed-stand-smote_time_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote --use_distances | tee "${FOLDER}baseline-speed+distances-stand-smote_time_f1-2x100.log"
