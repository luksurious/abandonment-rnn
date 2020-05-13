#!/usr/bin/env bash

COMMON_ARGS="--patience 8 --folds 10 --max_events 50 --min_events 2 --lr 1e-4 --layers 2 --units 100 --dropout 0.3 --batch_size 8"

# check varying strength
THIS_ARGS="--stop_val val_loss --aug_var_strength 1"
python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode vary  | tee "vary1_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode vary --use_time | tee "vary1-time_loss-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode vary+cut  | tee "vary1+cut_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode vary+cut --use_time | tee "vary1+cut-time_loss-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode varycutcomb  | tee "vary1+comb_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode varycutcomb --use_time | tee "vary1+comb-time_loss-2x100.log"

THIS_ARGS="--stop_val val_loss --aug_var_strength 3"
python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode vary  | tee "vary3_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode vary --use_time | tee "vary3-time_loss-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode vary+cut  | tee "vary3+cut_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode vary+cut --use_time | tee "vary3+cut-time_loss-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode varycutcomb  | tee "vary3+comb_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode varycutcomb --use_time | tee "vary3+comb-time_loss-2x100.log"

THIS_ARGS="--stop_val val_loss --aug_var_strength 5"
python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode vary  | tee "vary5_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode vary --use_time | tee "vary5-time_loss-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode vary+cut  | tee "vary5+cut_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode vary+cut --use_time | tee "vary5+cut-time_loss-2x100.log"

python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode varycutcomb  | tee "vary5+comb_loss-2x100.log"
python -u run.py $COMMON_ARGS $THIS_ARGS --aug_mode varycutcomb --use_time | tee "vary5+comb-time_loss-2x100.log"
