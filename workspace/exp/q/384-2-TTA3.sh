#!/bin/bash
cd /work2/i_kuroyanagi/kaggle/melanoma/workspace
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
python src/evaluate.py --config config/config384-2.yaml --fig_size 384 --No 2 --TTA 3 
EOF
) >exp/log/384-2-TTA3.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>exp/log/384-2-TTA3.log
  unset CUDA_VISIBLE_DEVICES.
fi
time1=`date +"%s"`
 ( python src/evaluate.py --config config/config384-2.yaml --fig_size 384 --No 2 --TTA 3  ) &>>exp/log/384-2-TTA3.log
ret=$?
sync || truetime2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>exp/log/384-2-TTA3.log
echo '#' Accounting: end_time=$time2 >>exp/log/384-2-TTA3.log
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/log/384-2-TTA3.log
echo '#' Finished at `date` with status $ret >>exp/log/384-2-TTA3.log
[ $ret -eq 137 ] && exit 100;
touch exp/q/done.21609
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH  --ntasks-per-node=1  -p shared --cpus-per-task 16 --ntasks-per-node=1  --open-mode=append -e exp/q/384-2-TTA3.log -o exp/q/384-2-TTA3.log  /work2/i_kuroyanagi/kaggle/melanoma/workspace/exp/q/384-2-TTA3.sh >>exp/q/384-2-TTA3.log 2>&1
