export cur_dir=`pwd`
export run_file="sup_syn22.py"
export save_exp_name="syn_sup_DATRS"

python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=22020 ${run_file} --save_root="${cur_dir}/exp/${save_exp_name}/" \
--lr=0.00005 --batch-size=4 --num_epochs=50 --backbone=DATRS \
\
--lamda 50 \
2>&1 \
| tee "log/${save_exp_name}.`date`"

