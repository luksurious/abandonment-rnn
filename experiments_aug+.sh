#!/usr/bin/env bash

COMMON_ARGS="--patience 10 --folds 10 --max_events 50 --min_events 2 --lr 1e-4 --layers 2 --units 100 --dropout 0.3 --batch_size 8"

THIS_ARGS="--stop_val val_f1_score --use_time"
#python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote | tee "aug+smote_time_f1-2x100.log"
#python -u run.py $COMMON_ARGS $THIS_ARGS --oversample adasyn | tee "aug+adasyn_time_f1-2x100.log"
#python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote --standardize | tee "aug+smote_time+stand_f1-2x100.log"
#python -u run.py $COMMON_ARGS $THIS_ARGS --oversample adasyn --standardize | tee "aug+adasyn_time+stand_f1-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote+ --standardize | tee "aug+smote+_time+stand_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote+ --standardize --aug_var_strength 3 | tee "aug+smote+-v3_time+stand_f1-2x100.log"

THIS_ARGS="--stop_val val_f1_score --use_time --aug_mode varycutcomb"
#python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote | tee "aug+smote_time_f1-2x100.log"
#python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote --standardize | tee "aug+smote_time+stand_f1-2x100.log"
#python -u run.py $COMMON_ARGS $THIS_ARGS --oversample adasyn | tee "aug+adasyn_time_f1-2x100.log"
#python -u run.py $COMMON_ARGS $THIS_ARGS --oversample adasyn --standardize | tee "aug+adasyn_time+stand_f1-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote+ --standardize | tee "aug+smote+_time+stand_f1-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --oversample smote+ --standardize --aug_var_strength 3 | tee "aug+smote+-v3_time+stand_f1-2x100.log"
