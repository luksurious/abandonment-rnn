#!/usr/bin/env bash

# test 1x100
python -u run.py --patience 5 --stop_val val_loss --folds 10 --max_events 50 --min_events 2 --lr 3e-4 --layers 1 --units 100 --dropout 0.5 --batch_size 8 --use_time  | tee "model-1x100-time-b8-loss.log"

# test 1x50
python -u run.py --patience 5 --stop_val val_loss --folds 10 --max_events 50 --min_events 2 --lr 3e-4 --layers 1 --units 50 --dropout 0.5 --batch_size 8  --use_time | tee "model-1x50-time-b8-loss.log"



# test 2x100
python -u run.py --patience 5 --stop_val val_loss --folds 10 --max_events 50 --min_events 2 --lr 3e-4 --layers 2 --units 100 --dropout 0.3 --batch_size 8 --use_time  | tee "model-2x100-time-b8-loss.log"

# test 2x50
python -u run.py --patience 5 --stop_val val_loss --folds 10 --max_events 50 --min_events 2 --lr 3e-4 --layers 2 --units 50 --dropout 0.3 --batch_size 8  --use_time | tee "model-2x50-time-b8-loss.log"


# test 3x100
python -u run.py --patience 5 --stop_val val_loss --folds 10 --max_events 50 --min_events 2 --lr 3e-4 --layers 3 --units 100 --dropout 0.2 --batch_size 8 --use_time  | tee "model-3x100-time-b8-loss.log"

# test 3x50
python -u run.py --patience 5 --stop_val val_loss --folds 10 --max_events 50 --min_events 2 --lr 3e-4 --layers 3 --units 50 --dropout 0.2 --batch_size 8  --use_time | tee "model-3x50-time-b8-loss.log"

