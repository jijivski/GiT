GPU_NUM=1
work_dir=/224045019/6051_final_project/GiT/work
# support_num=10
# export MMENGINE_LAZY_IMPORT=1
# 关闭 lazy import
# export MMENGINE_LAZY_IMPORT=0
bash tools/dist_train.sh configs/GiT/few-shot/few_shot_isic.py ${GPU_NUM} --work-dir ${work_dir}

# bash tools/dist_train.sh configs/GiT/few-shot/few_shot_isic.py ${GPU_NUM} --work-dir ${work_dir} --cfg-options support_num=${support_num}

# put text here