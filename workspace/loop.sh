#!/bin/bash
. ./path.sh || exit 1
. ./cmd.sh || exit 1

. utils/parse_options.sh || exit 1
for No in 0 1 2; do
  for fig_size in 256 384 512 768; do
    echo "fig_size:${fig_size}, No:${No}"
    sbatch -c 4 -n 4 --gres=gpu:1 -J "${fig_size}-${No}" ./run.sh --fig_size ${fig_size} --No ${No}
  done
done
