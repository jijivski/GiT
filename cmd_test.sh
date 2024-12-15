
GPU_NUM=2


# ckpt_file='work/det_few_shot_isic_support_200_iter_20.pth'
# +-------------+-------+-------+-------+
# |    Class    |  IoU  |  Acc  |  Dice |
# +-------------+-------+-------+-------+
# | skin_lesion | 76.14 | 86.45 | 86.45 |
# +-------------+-------+-------+-------+
# ckpt_file='work/det_few_shot_isic_support_200_iter_50.pth'
# +-------------+-------+-------+-------+
# |    Class    |  IoU  |  Acc  |  Dice |
# +-------------+-------+-------+-------+
# | skin_lesion | 80.13 | 88.97 | 88.97 |
# +-------------+-------+-------+-------+
# ckpt_file='work/det_few_shot_isic_support_200_iter_100.pth'
# +-------------+------+-------+-------+
# |    Class    | IoU  |  Acc  |  Dice |
# +-------------+------+-------+-------+
# | skin_lesion | 84.9 | 91.83 | 91.83 |
# +-------------+------+-------+-------+


# ckpt_file='work/det_few_shot_isic_support_1000_iter_20.pth'
# +-------------+-------+-------+-------+
# |    Class    |  IoU  |  Acc  |  Dice |
# +-------------+-------+-------+-------+
# | skin_lesion | 78.68 | 88.07 | 88.07 |
# +-------------+-------+-------+-------+
# ckpt_file='work/det_few_shot_isic_support_1000_iter_50.pth'
# +-------------+-------+-------+-------+
# |    Class    |  IoU  |  Acc  |  Dice |
# +-------------+-------+-------+-------+
# | skin_lesion | 86.06 | 92.51 | 92.51 |
# +-------------+-------+-------+-------+
# ckpt_file='work/det_few_shot_isic_support_1000_iter_100.pth'
# +-------------+-------+-------+-------+
# |    Class    |  IoU  |  Acc  |  Dice |
# +-------------+-------+-------+-------+
# | skin_lesion | 86.27 | 92.63 | 92.63 |
# +-------------+-------+-------+-------+





work_dir=/224045019/6051_final_project/GiT/work

# bash tools/dist_test.sh configs/GiT/few-shot/few_shot_isic.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
bash tools/dist_test.sh configs/GiT/zero-shot/zero_shot_isic_seg.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}