#!/usr/bin/env bash

COMMON_ARGS="--patience 5 --stop_val val_loss --folds 10 --max_events 50 --min_events 2 --lr 1e-4 --layers 2 --units 100 --dropout 0.3 --batch_size 8 --all_aband"

FOLDER="experiments/allaband-"

# Standardized baselines
THIS_ARGS="--no_augment --standardize"
#python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "${FOLDER}baseline-stand-undersample_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "${FOLDER}baseline-stand-undersample_time_loss-2x100.log"

#python -u run.py $COMMON_ARGS $THIS_ARGS --oversample random  | tee "${FOLDER}baseline-stand-oversample_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample random --use_time | tee "${FOLDER}baseline-stand-oversample_time_loss-2x100.log"

#python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote  | tee "${FOLDER}baseline-stand-smote_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote --use_time | tee "${FOLDER}baseline-stand-smote_time_loss-2x100.log"

#python -u run.py $COMMON_ARGS $THIS_ARGS --oversample adasyn  | tee "${FOLDER}baseline-stand-adasyn_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample adasyn --use_time | tee "${FOLDER}baseline-stand-adasyn_time_loss-2x100.log"


# Custom runs
THIS_ARGS="--standardize"
#python -u run.py $COMMON_ARGS $THIS_ARGS  | tee "${FOLDER}custom-stand_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --use_time | tee "${FOLDER}custom-stand_time_loss-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --use_time --no_coords | tee "${FOLDER}custom-stand_time_loss-2x100.log"
