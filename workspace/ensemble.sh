#!/bin/bash
. ./path.sh || exit 1
. ./cmd.sh || exit 1
n_cpu=16
. utils/parse_options.sh || exit 1
source /home/i_kuroyanagi/workspace/virtualenvs/asdpy36/bin/activate

python src/stacking.py
# kaggle competitions submit -c siim-isic-melanoma-classification -f exp/ensemble/submission_xgb.csv -m "Message"