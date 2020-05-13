#!/usr/bin/env bash

COMMON_ARGS="--patience 5 --stop_val val_f1_score --folds 10 --max_events 50 --min_events 2 --layers 2 --units 100 --dropout 0.3 --standardize --use_time --lr=1e-4 --batch_size 4"

#python -u run.py $COMMON_ARGS --batch_size 8 | tee "best-b8.log"
#python -u run.py $COMMON_ARGS --batch_size 4 | tee "best-b4.log"
#python -u run.py $COMMON_ARGS --batch_size 16 | tee "best-b16.log"
#
#
#python -u run.py $COMMON_ARGS --batch_size 8 --lr 1e-4 | tee "best-lr1e-4.log"
#python -u run.py $COMMON_ARGS --batch_size 8 --lr 1e-3 | tee "best-lr1e-3.log"
#python -u run.py $COMMON_ARGS --batch_size 8 --lr 1e-5 | tee "best-lr1e-5.log"
#
#
#python -u run.py $COMMON_ARGS --batch_size 8 --optimizer RMSprop | tee "best-rmsprop.log"
#python -u run.py $COMMON_ARGS --batch_size 8 --optimizer Nadam | tee "best-nadam.log"
#python -u run.py $COMMON_ARGS --batch_size 8 --optimizer SDG | tee "best-sdg.log"


#python -u run.py $COMMON_ARGS --aug_var_strength 10 | tee "best-vary10.log"
python -u run.py $COMMON_ARGS --aug_var_strength 3 | tee "best-vary3.log"
#python -u run.py $COMMON_ARGS --aug_var_strength 1 | tee "best-vary1.log"
#python -u run.py $COMMON_ARGS --aug_varycount 2 | tee "best-varycount2.log"
python -u run.py $COMMON_ARGS --aug_varycount 4 | tee "best-varycount4.log"
#python -u run.py $COMMON_ARGS --aug_cutoff_limit 4 | tee "best-aug_cutoff_limit4.log"
python -u run.py $COMMON_ARGS --aug_cutoff_limit 3 | tee "best-aug_cutoff_limit3.log"
#python -u run.py $COMMON_ARGS --aug_cutoff_limit 10 | tee "best-aug_cutoff_limit10.log"
python -u run.py $COMMON_ARGS --aug_varycount 4 --aug_cutoff_limit 3 --aug_var_strength 3 | tee "best-cutoff3+varyc4+varys3.log"
python -u run.py $COMMON_ARGS --aug_varycount 4 --aug_cutoff_limit 3 --aug_var_strength 3 --aug_mode varycutcomb | tee "best-cutoff3+varyc4+varys3-comb.log"

