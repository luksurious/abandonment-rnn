# Query Abandonment Prediction with Recurrent Neural Models of Mouse Cursor Movements

Experiment code for the [paper](https://doi.org/10.1145/3340531.3412126) by [Lukas Brückner](https://github.com/luksurious/), [Ioannis Arapakis](https://iarapakis.github.io/), and [Luis A. Leiva](https://luis.leiva.name/web/)

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

The data used is from [Predicting User Engagement with Direct Displays Using Mouse Cursor Information](https://dl.acm.org/doi/10.1145/2911451.2911505).
Unfortunately, it is proprietary, so it cannot be shared in this repository.

## Citation

Please use the following Bibtex when citing this work.

```
@InProceedings{Bruckner20_abandonment,
  author    = {Lukas Brückner and Ioannis Arapakis and Luis A. Leiva},
  title     = {Query Abandonment Prediction with Deep Learning Models of Mouse Cursor Movements},
  booktitle = {Proc. CIKM},
  year      = {2020},
  doi       = {10.1145/3340531.3412126},
}
```
