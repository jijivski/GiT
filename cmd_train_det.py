import os
import glob
import subprocess

gpu_num = 2
work_dir = r'/224045019/6051_final_project/GiT/work'
# support_num_list = [10, 50, 100, 200, 500, 1000]

support_num_list = [100, 200, 500, 1000, 2000]
for support_num in support_num_list:

    # 为每个实验创建独立的工作目录
    # work_dir = os.path.join(base_work_dir, f'support_{support_num}')
    # os.makedirs(work_dir, exist_ok=True)
    
    # 使用环境变量传递参数
    # my_env = os.environ.copy()
    # my_env['SUPPORT_NUM'] = str(support_num)
    
    # cmd = f'bash tools/dist_train.sh configs/GiT/few-shot/few_shot_isic.py {gpu_num} --work-dir {work_dir} --cfg-options support_num={support_num}'
    cmd = f'bash tools/dist_train.sh configs/GiT/few-shot/few_shot_isic_{support_num}.py {gpu_num} --work-dir {work_dir}'

    try:
        # subprocess.run(cmd, shell=True, check=True, env=my_env)
        subprocess.run(cmd, shell=True, check=True, )
        
    except subprocess.CalledProcessError as e:
        print(f'训练命令执行失败: {e}')
        breakpoint()
        continue
    # 查找所有checkpoint文件

    try:
        ckpt_files = glob.glob(os.path.join(work_dir, 'iter_*.pth'))
        if not ckpt_files:
            print(f'未找到checkpoint文件在 {work_dir}')
            continue
            
        # 重命名所有文件
        for ckpt_file in ckpt_files:
            iter_num = ckpt_file.split('_')[-1].split('.')[0]
            new_name = os.path.join(work_dir, f'det_few_shot_isic_support_{support_num}_iter_{iter_num}.pth')
            os.rename(ckpt_file, new_name)
            print(f'成功重命名checkpoint文件为: {new_name}')
        
    except Exception as e:
        print(f'处理checkpoint文件时出错: {e}')
        breakpoint()


'''
cd /224045019/6051_final_project/GiT/
python cmd_train_det.py
'''