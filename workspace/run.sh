#!/bin/bash
. ./path.sh || exit 1
. ./cmd.sh || exit 1

n_cpu=16
n_gpu=1
No=0
fig_size=256
stage=0
stop_stage=10
. utils/parse_options.sh || exit 1

config="config/config${fig_size}-${No}.yaml"
# log_dir="exp/${fig_size}-${No}"
echo "config:${config}"
echo "No:${No}, fig_size:${fig_size}, stage:${stage}, stop_stage:${stop_stage}"
echo "n_cpu:${n_cpu}, n_gpu:${n_gpu}, train_cmd:${train_cmd}"

source /home/i_kuroyanagi/workspace/virtualenvs/asdpy36/bin/activate

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Traingin model"
    # ${train_cmd} --num_threads ${n_cpu} --gpu ${n_gpu} --p hpc "${log_dir}/train.log" \
    python src/train2.py \
        --config ${config} \
        --fig_size ${fig_size} \
        --No ${No} \
        --pass_list 1 2 3
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Evaluate model"
    # ${train_cmd} --num_threads ${n_cpu} --gpu ${n_gpu} --p hpc "${log_dir}/train.log" \
    python src/evaluate.py \
        --config ${config} \
        --fig_size ${fig_size} \
        --No ${No}
fi
# sbatch -c 4 -n 4 --gres=gpu:1 -J 384-2 ./run.sh --fig_size 384 --No 2
# sbatch -J 384 ./run.sh --fig_size 384
# sbatch -c 4 -n 4 --gres=gpu:1 -J 256-2 ./run.sh --fig_size 256 --No 2 --stage 1
