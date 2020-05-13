#!/usr/bin/env bash

COMMON_ARGS="--patience 10 --stop_val val_loss --folds 10 --max_events 50 --min_events 2 --lr 3e-4 --layers 2 --units 100 --dropout 0.3 --batch_size 8"

# Raw baselines
THIS_ARGS="--no_augment"
python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "baseline-undersample_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "baseline-undersample_time_loss-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --oversample random  | tee "baseline-oversample_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample random --use_time | tee "baseline-oversample_time_loss-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote  | tee "baseline-smote_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote --use_time | tee "baseline-smote_time_loss-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --oversample adasyn  | tee "baseline-adasyn_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample adasyn --use_time | tee "baseline-adasyn_time_loss-2x100.log"


# Standardized baselines
THIS_ARGS="--no_augment --standardize"
python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "baseline-stand-undersample_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "baseline-stand-undersample_time_loss-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --oversample random  | tee "baseline-stand-oversample_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample random --use_time | tee "baseline-stand-oversample_time_loss-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote  | tee "baseline-stand-smote_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote --use_time | tee "baseline-stand-smote_time_loss-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --oversample adasyn  | tee "baseline-stand-adasyn_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample adasyn --use_time | tee "baseline-stand-adasyn_time_loss-2x100.log"


# Custom runs
THIS_ARGS=""
python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "custom_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "custom_time_loss-2x100.log"

THIS_ARGS="--standardize"
python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "custom-stand_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "custom-stand_time_loss-2x100.log"

THIS_ARGS="--aug_no_balanced"
python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "custom-no-balance_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "custom-no-balance_time_loss-2x100.log"

THIS_ARGS="--aug_no_balanced --standardize"
python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "custom-no-balance-stand_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "custom-no-balance-stand_time_loss-2x100.log"


THIS_ARGS="--aug_mode varycutcomb"
python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "custom-augcomb_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "custom-augcomb_time_loss-2x100.log"


THIS_ARGS="--aug_mode vary"
python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "custom-varyonly_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "custom-varyonly_time_loss-2x100.log"

THIS_ARGS="--aug_mode cut"
python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "custom-cutonly_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "custom-cutonly_time_loss-2x100.log"


