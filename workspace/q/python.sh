#!/bin/bash
cd /work2/i_kuroyanagi/kaggle/melanoma/workspace
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
-m src/train2.py --config config/config256-0.yaml --fig_size 256 --No 0 
EOF
) >python
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>python
  unset CUDA_VISIBLE_DEVICES.
fi
time1=`date +"%s"`
 ( -m src/train2.py --config config/config256-0.yaml --fig_size 256 --No 0  ) &>>python
ret=$?
sync || truetime2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>python
echo '#' Accounting: end_time=$time2 >>python
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>python
echo '#' Finished at `date` with status $ret >>python
[ $ret -eq 137 ] && exit 100;
touch ./q/done.641
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH  --ntasks-per-node=1  -p gpu --gres=gpu:1 --time 4:0:0 --cpus-per-task 16 --ntasks-per-node=1  --open-mode=append -e ./q/python -o ./q/python  /work2/i_kuroyanagi/kaggle/melanoma/workspace/./q/python.sh >>./q/python 2>&1
