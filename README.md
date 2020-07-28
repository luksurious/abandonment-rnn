# Query Abandonment Prediction with Recurrent Neural Models of Mouse Cursor Movements

Experiment code for the paper by [Lukas Brückner](https://github.com/luksurious/), [Ioannis Arapakis](https://iarapakis.github.io/), and [Luis A. Leiva](https://luis.leiva.name/web/)

## Requirements

* Python 3
* TensorFlow 2
* _and more as listed in requirements.txt_

Install all required packages in a virtual/conda/... environment with `pip install -r requirements.txt`

## Running

Entry point is always `run.py`.
Parameters of the experiments and their execution are listed in `experiments.sh`.

_Examples_: 

* `python run.py --max_events 50 --min_events 2 --all_bad` (simple minority voting)
* `python run.py --max_events 50 --min_events 2 --standardize --patience 5 --stop_val val_f1_score --folds 10 --lr 1e-4 --layers 2 --units 100 --dropout 0.3 --batch_size 4 --use_time` (final parameters for custom augmentation with RNN model)

## Data

Unfortunately, the data we used is proprietary, so it cannot be shared in this repository.
