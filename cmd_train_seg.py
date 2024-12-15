import os
import glob
import subprocess
import json
gpu_num = 2
work_dir = r'/224045019/6051_final_project/GiT/work'

glob_file_path = r'/224045019/6051_final_project/GiT/work/det_few_shot_isic_support_*.pth'
ckpt_files = glob.glob(glob_file_path)
ckpt_files = sorted(ckpt_files)

support_num_list = [10, 100, 200, 500,]
# support_num_list = [100,500]

_cnt = 0
for support_num in support_num_list:
    for ckpt_file in ckpt_files:
        # 从文件名中提取support_num和iter_num
        ckpt_basename = os.path.basename(ckpt_file)
        # breakpoint()    
        if ckpt_basename=='det_few_shot_isic_support_1000_iter_50.pth':
            continue    
        
        ckpt_support_num = int(ckpt_basename.split('_')[5])
        ckpt_iter_num = int(ckpt_basename.split('_')[-1].split('.')[0])
        if ckpt_iter_num > 50:
            print(f'skip {ckpt_iter_num}')
            continue
        
        # 读取配置文件
        with open(r'/224045019/6051_final_project/GiT/configs/GiT/few-shot/few_shot_isic_seg.py', 'r', encoding='utf-8') as f:
            config_content = f.read()

        # 替换前两行
        config_lines = config_content.split('\n')
        # ckpt_file = './universal_base.pth'
        config_lines[0] = f'load_from = "{ckpt_file}"'
        config_lines[1] = f'support_num={support_num}'
        config_content = '\n'.join(config_lines)

        # 创建实验特定的配置文件
        exp_config_path = f'configs/GiT/few-shot/few_shot_isic_seg_template.py'
        with open(exp_config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)

        _cnt += 1
        # 34 / {'support_num': 100, 'ckpt_support_num': 200, 'ckpt_iter_num': 90}
        dict_str = {'support_num': support_num, 'ckpt_support_num': ckpt_support_num, 'ckpt_iter_num': ckpt_iter_num}

        # 检查是否已经存在相同的实验记录
        skip_experiment = False

        # 先读取现有内容
        try:
            with open('cmd_train_seg_rec.jsonl', 'r', encoding='utf-8') as f:
                existing_records = f.read().strip().split('\n')
                for record in existing_records:
                    if record and eval(record) == dict_str:
                        skip_experiment = True
                        break
        except FileNotFoundError:
            existing_records = []

        # 如果不存在相同记录，则添加新记录
        if not skip_experiment:


            print(f'{_cnt} / {dict_str}')
            cmd = f'bash tools/dist_train.sh configs/GiT/few-shot/few_shot_isic_seg_template.py {gpu_num} --work-dir {work_dir} > ./seg_train_logs/seg_train_log_{ckpt_support_num}_{ckpt_iter_num}_{support_num}_full.txt'
            # breakpoint()
            print(cmd)
            # input()
            try:
                subprocess.run(cmd, shell=True, check=True)
                
            except subprocess.CalledProcessError as e:
                print(f'训练命令执行失败: {e}')
                breakpoint()
                continue
            
            with open('cmd_train_seg_rec.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(dict_str) + '\n')
        else:
            print(f'skip {dict_str}')


'''
cd /224045019/6051_final_project/GiT/
python cmd_train_seg.py > train_seg_1213.log
'''