GPU_NUM=1
work_dir=/224045019/6051_final_project/GiT/work

bash tools/dist_train.sh configs/GiT/few-shot/few_shot_isic_seg.py ${GPU_NUM} --work-dir ${work_dir}

# put text here
