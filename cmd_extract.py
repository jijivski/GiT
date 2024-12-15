# get a df of all the seg train logs
import pandas as pd
import glob
import re
import matplotlib.pyplot as plt
import os
import numpy as np
def extract_params_from_log(log_path):
    # 从日志文件中提取参数
    with open(log_path, 'r') as f:
        content = f.readlines()
        
    support_num = None
    ckpt_path = None
    
    for line in content:
        # 提取support_num
        if 'support_num = ' in line:
            support_num = int(re.findall(r'support_num = (\d+)', line)[0])
            
        # 提取checkpoint路径
        if 'Load checkpoint from' in line:
            ckpt_path = line.split('Load checkpoint from ')[1].strip()
            if ckpt_path:
                # 从checkpoint路径提取参数
                match = re.search(r'support_(\d+)_iter_(\d+)', ckpt_path)
                if match:
                    ckpt_support = int(match.group(1))
                    ckpt_iter = int(match.group(2))
                    return {
                        'support_num': support_num,
                        'ckpt_support_num': ckpt_support,
                        'ckpt_iter_num': ckpt_iter
                    }
    return None

def extract_support_num(log_path):
    # 从日志文件中提取参数
    with open(log_path, 'r') as f:
        content = f.readlines()
    for line in content:
        # 提取support_num
        if 'support_num = ' in line:
            support_num = int(re.findall(r'support_num = (\d+)', line)[0])
            return support_num
    return None


def extract_metrics_from_content(file_path):
    # 从文件内容中提取指标数据
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 使用正则表达式提取所有匹配的指标数据
    metrics = re.findall(r'skin_lesion\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)', content)
    
    if metrics:
        # 为每个时间点创建一个结果
        results = []
        for i, m in enumerate(metrics):
            result = {
                'step': i*10,
                'IoU': float(m[0]),
                'Acc': float(m[1]),
                'Dice': float(m[2])
            }
            results.append(result)
        return results
    return None

# 获取work目录下的所有日志文件
base_dir = '/224045019/6051_final_project/GiT/work'
# target_time = '20241208_092137' # kmeans 
# target_time = '20241213_072254'# raw detail
target_time  = '20241213_142519'
results = []

_cnt = 0
# 遍历work目录
for dirname in os.listdir(base_dir):
    if dirname > target_time:  # 只处理大于目标时间的文件夹
        log_path = os.path.join(base_dir, dirname, f'{dirname}.log')
        if os.path.exists(log_path):
            _cnt += 1
            params = extract_params_from_log(log_path)
            metrics_list = extract_metrics_from_content(log_path)
            print(f'{_cnt} / {params}')
            if params and metrics_list:
                for metrics in metrics_list:
                    result = {**params, **metrics}
                    results.append(result)

# 从指定文件夹提取基准参数
base_log_path = '/224045019/6051_final_project/GiT/seg_train_logs/from_raw_all_detail/'
for dirname in os.listdir(base_log_path):
    log_path = os.path.join(base_log_path, f'{dirname}')
    if os.path.exists(log_path):
        base_params = extract_params_from_log(log_path)
        if not base_params:
            support_num=extract_support_num(log_path)
            
            if support_num:
                base_params={'support_num':support_num,
                            'ckpt_support_num': 0,
                            'ckpt_iter_num': 0
                            }
                metrics_list = extract_metrics_from_content(log_path)
                _cnt+=1
                print(f'{_cnt} / {base_params}')
                if base_params and metrics_list:
                    for metrics in metrics_list:
                        result = {**base_params, **metrics}
                        results.append(result)
            else:
                #even can not find support num
                print(f'fail to find anything about training para in {log_path}')
                continue
                

print(f'total {_cnt} logs')
# 创建数据框
df = pd.DataFrame(results)
print(df)

df.sort_values(by=['ckpt_support_num', 'ckpt_iter_num', 'support_num','step'], inplace=True, ascending=[True, True, True, True])
df.to_csv('seg_train_log_df.csv', index=False)
# 1. 固定ckpt_support_num和ckpt_iter_num，观察step的变化
def plot_metrics_by_step(df, ckpt_support, ckpt_iter):
    data = df[(df['ckpt_support_num'] == ckpt_support) & 
              (df['ckpt_iter_num'] == ckpt_iter) &
              (df['support_num'] == 100)]
    
    plt.figure(figsize=(10, 6))
    
    # 使用移动平均进行平滑处理
    window_size = 1
    iou_smooth = data['IoU'].rolling(window=window_size, center=True).mean()
    dice_smooth = data['Dice'].rolling(window=window_size, center=True).mean()
    
    # 绘制原始数据点
    # plt.plot(data['step'], data['IoU'], 'o', alpha=0.3, label='IoU (raw)')
    # plt.plot(data['step'], data['Dice'], '^', alpha=0.3, label='Dice (raw)')
    
    # 绘制平滑曲线
    plt.plot(data['step'], iou_smooth, '--', label='IoU')
    plt.plot(data['step'], dice_smooth, '-', label='Dice')
    
    plt.title(f'indicator change with step')
    plt.xlabel('Step')
    plt.ylabel('indicator')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'figures/metrics_change_with_step_ckpt_support_{ckpt_support}_ckpt_iter_{ckpt_iter}.png')

# 1. 固定ckpt_support_num和ckpt_iter_num,step，support_num， 观察的变化
def plot_metrics_by_ckpt(df):
    # 按照三个变量分组
    # select step==90
    df = df[df['step'] == 90]   
    grouped = df.groupby(['ckpt_support_num', 'ckpt_iter_num', 'step'])
    
    plt.figure(figsize=(15, 8))
    
    # 为每个组绘制一条线
    for name, group in grouped:
        ckpt_support, ckpt_iter, step = name
        label = f'support={ckpt_support},iter={ckpt_iter},step={step}'
        plt.plot(group['support_num'], group['IoU'], marker='o', label=label)
    
    plt.title(f'IoU change with support_num (ckpt_support={ckpt_support}, ckpt_iter={ckpt_iter}, step={step})')
    plt.xlabel('support_num')
    plt.ylabel('IoU')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig('figures/all_groups_metrics_change.png', bbox_inches='tight')

# 2. 固定step 和ckpt_iter，比较不同ckpt_support_num的效果
def plot_metrics_by_ckpt_support(df, step_value, ckpt_iter):
    data = df[(df['step'] == step_value) & 
              (df['ckpt_iter_num'] == ckpt_iter)]
    
    plt.figure(figsize=(12, 6))
    metrics = ['IoU', 'Acc', 'Dice']
    markers = ['o', 's', '^']
    
    for metric, marker in zip(metrics, markers):
        plt.plot(data['ckpt_support_num'], data[metric], 
                marker=marker, label=metric, linestyle='-')
    
    plt.title(f'Metrics change with ckpt_support_num (step={step_value}, ckpt_iter={ckpt_iter})')
    plt.xlabel('ckpt_support_num')
    plt.ylabel('Metrics Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'figures/metrics_change_with_ckpt_support_num_step_{step_value}.png')

# def plot_metrics_by_base_model(df, seg_support_num):
#     data = df[(df['support_num'] == seg_support_num) & (df['step'] == 90)]
#     ckpt_supports = sorted(data['ckpt_support_num'].unique())
    
#     plt.figure(figsize=(15, 8))
#     bar_width = 0.2
    
#     for i, ckpt_support in enumerate(ckpt_supports):
#         ckpt_data = data[data['ckpt_support_num'] == ckpt_support].sort_values('ckpt_iter_num')
#         iters = ckpt_data['ckpt_iter_num'].unique()
#         x = np.arange(len(iters))
        
#         plt.bar(x + i*bar_width, 
#                ckpt_data['Acc'], 
#                width=bar_width, 
#                label=f'ckpt_support={ckpt_support}')
    
#     plt.title(f'IoU for Different Base Models (seg_support={seg_support_num})')
#     plt.xlabel('ckpt_iter_num')
#     plt.ylabel('IoU')
#     plt.legend()
#     plt.xticks(x + bar_width*(len(ckpt_supports)-1)/2, 
#                [str(iter_num) for iter_num in iters])
#     plt.grid(True, axis='y')
#     plt.tight_layout()
#     plt.savefig(f'figures/base_model_comparison_seg_support_{seg_support_num}.png')
#     plt.show()

def plot_metrics_by_base_model(df, seg_support_num,step=30):
    data = df[(df['support_num'] == seg_support_num) & (df['step'] == step)]
    ckpt_iter_nums = sorted(data['ckpt_iter_num'].unique())
    
    plt.figure(figsize=(15, 8))
    bar_width = 0.02
    
    for i, ckpt_iter_num in enumerate(ckpt_iter_nums):
        ckpt_data = data[data['ckpt_iter_num'] == ckpt_iter_num].sort_values('ckpt_support_num')
        supports = ckpt_data['ckpt_support_num'].unique()
        x = np.arange(len(supports))
        
        plt.bar(x + i*bar_width, 
               ckpt_data['Acc'], 
               width=bar_width, 
               label=f'ckpt_iter_num={ckpt_iter_num}')
    
    plt.title(f'IoU for Different Base Models (seg_support={seg_support_num},step={step})')
    plt.xlabel('ckpt_iter_num')
    plt.ylabel('IoU')
    # 图例放到外面
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.xticks(x + bar_width*(len(ckpt_iter_nums)-1)/2, 
    #            [str(iter_num) for iter_num in ckpt_iter_nums])
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f'figures/base_model_comparison_seg_support_{seg_support_num}_step_{step}_group_by_ckpt_support_num.png')
    plt.show()

def plot_metrics_by_seg_support(df, ckpt_support, ckpt_iter,step=20):
    # 固定基座，比较不同seg support的表现
    data = df[(df['ckpt_support_num'] == ckpt_support) & 
              (df['ckpt_iter_num'] == ckpt_iter) & 
              (df['step'] == step)]
    
    # 获取所有唯一的support_num值
    supports = sorted(data['support_num'].unique())
    
    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    metrics = ['IoU', 'Acc', 'Dice']
    
    x = np.arange(len(metrics))
    for i, support in enumerate(supports):
        support_data = data[data['support_num'] == support]
        values = [support_data[metric].iloc[0] for metric in metrics]
        plt.bar(x + i*bar_width, values, 
               width=bar_width, 
               label=f'seg_support={support}')
    
    plt.title(f'Metrics Comparison (ckpt_support={ckpt_support}, ckpt_iter={ckpt_iter},step={step})')
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.legend()
    plt.xticks(x + bar_width*(len(supports)-1)/2, metrics)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f'figures/seg_support_comparison_ckpt_{ckpt_support}_{ckpt_iter}_step_{step}.png')
    plt.show()



# 调用新函数的示例
# for step in [30,60,90]:
#     plot_metrics_by_base_model(df, seg_support_num=100,step=step)  # 固定seg_support=100,和setp， 比较不同基座，（能不能再来一个最大值做trick？）

# for ckpt_support in [200,500,1000]:
#     plot_metrics_by_seg_support(df, ckpt_support=ckpt_support, ckpt_iter=20)  # 固定基座ckpt_support=500, ckpt_iter=40

# # !pip install seaborn
import seaborn as sns
# # 创建热力图显示不同support组合下的最大性能
# def plot_heatmap(df,metric='IoU'):
    
#     # 获取唯一的support_num和ckpt_support_num值
#     support_nums = sorted(df['support_num'].unique())
#     ckpt_supports = sorted(df['ckpt_support_num'].unique())
    
#     # 创建空矩阵存储最大IoU值
#     heatmap_data = np.zeros((len(support_nums), len(ckpt_supports)))
    
#     # 填充矩阵数据
#     for i, support in enumerate(support_nums):
#         for j, ckpt_support in enumerate(ckpt_supports):
#             # 获取该组合下的所有IoU值
#             iou_values = df[(df['support_num'] == support) & 
#                           (df['ckpt_support_num'] == ckpt_support)][metric]
#             if len(iou_values) > 0:
#                 heatmap_data[i,j] = iou_values.max()
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(heatmap_data, 
#                 xticklabels=ckpt_supports,
#                 yticklabels=support_nums,
#                 annot=True, 
#                 fmt='.2f',
#                 cmap='YlOrRd')
    
#     plt.title('Heatmap of different support combinations')
#     plt.xlabel('Checkpoint Support Number')
#     plt.ylabel('Segmentation Support Number')
#     plt.tight_layout()
#     plt.savefig(f'figures/support_heatmap_{metric}.png')
#     plt.show()

# # 调用热力图函数
# metrics = ['IoU', 'Acc', 'Dice']
# for metric in metrics:
#     plot_heatmap(df,metric)
# 筛选出ckpt_iter=50和step=30的数据
# df_filtered = df[(df['ckpt_iter_num'] == 50) & (df['step'] == 30)]


def draw_heatmap(df, fixed_ckpt_iter, fixed_step): 
    df_filtered = df[(df['ckpt_iter_num'] == fixed_ckpt_iter) & (df['step'] == fixed_step)]
    if df_filtered.empty:
        print(f"No data found for ckpt_iter={fixed_ckpt_iter} and step={fixed_step}")
        return
    # 创建一个新的图形，指定大小
    plt.figure(figsize=(15,3))
    
    metrics = ['IoU', 'Dice']
    for idx, metric in enumerate(metrics, 1):
        # 创建子图
        ax = plt.subplot(1, 2, idx)
        
        # 获取唯一的support_num和ckpt_support_num值
        support_nums = sorted(df_filtered['support_num'].unique())
        ckpt_supports = sorted(df_filtered['ckpt_support_num'].unique())
        
        # 创建空矩阵存储性能值
        heatmap_data = np.zeros((len(support_nums), len(ckpt_supports)))
        
        # 填充矩阵数据
        for i, support in enumerate(support_nums):
            for j, ckpt_support in enumerate(ckpt_supports):
                values = df_filtered[(df_filtered['support_num'] == support) & 
                                   (df_filtered['ckpt_support_num'] == ckpt_support)][metric]
                if len(values) > 0:
                    heatmap_data[i,j] = values.iloc[0]
        
        # 绘制热力图
        sns.heatmap(heatmap_data, 
                   xticklabels=ckpt_supports,
                   yticklabels=support_nums,
                   annot=True, 
                   fmt='.2f',
                   cmap='Blues',
                   vmin=80,  
                   vmax=95,  
                   ax=ax)
        
        ax.set_title(f'Heatmap of {metric}')
        ax.set_xlabel('Checkpoint Support Number')
        ax.set_ylabel('Segmentation Support Number')
    plt.tight_layout()
    plt.savefig(f'figures/heatmaps/support_heatmap_metrics_ckptiter{fixed_ckpt_iter}_step{fixed_step}.png')
    plt.clf()  # 关闭图形以释放内存





def draw_heatmap_average(df, use='mean'): 
    # 计算平均值版本
    if use == 'mean':
        # df_filtered_mean = df[(df['ckpt_iter_num'] == fixed_ckpt_iter) & (df['step'] == fixed_step)]
        df_filtered_mean = df.groupby(['support_num', 'ckpt_support_num']).agg({
            'IoU': 'mean',
            'Acc': 'mean', 
            'Dice': 'mean'
        }).reset_index()
        df_filtered=df_filtered_mean

    elif use == 'median':
        # df_filtered_median = df[(df['ckpt_iter_num'] == fixed_ckpt_iter) & (df['step'] == fixed_step)]
        df_filtered_median = df.groupby(['support_num', 'ckpt_support_num']).agg({
            'IoU': 'median',
            'Acc': 'median',
            'Dice': 'median' 
        }).reset_index()
        df_filtered=df_filtered_median

    elif use == 'max':
        # df_filtered_median = df[(df['ckpt_iter_num'] == fixed_ckpt_iter) & (df['step'] == fixed_step)]
        df_filtered_median = df.groupby(['support_num', 'ckpt_support_num']).agg({
            'IoU': 'max',
            'Acc': 'max',
            'Dice': 'max' 
        }).reset_index()
        df_filtered=df_filtered_median


    # if df_filtered_mean.empty or df_filtered_median.empty:
    #     print(f"No data found for ckpt_iter={fixed_ckpt_iter} and step={fixed_step}")
    #     return

    # 创建两个图形,一个用于平均值,一个用于中位数
    plt.figure(figsize=(12,3))
    # plt.figure(figsize=(15, 12))
    
    metrics = ['IoU', 'Dice']
    for idx, metric in enumerate(metrics, 1):
        # 创建子图
        ax = plt.subplot(1, 2, idx)
        
        # 获取唯一的support_num和ckpt_support_num值
        support_nums = sorted(df_filtered['support_num'].unique())
        ckpt_supports = sorted(df_filtered['ckpt_support_num'].unique())
        
        # 创建空矩阵存储性能值
        heatmap_data = np.zeros((len(support_nums), len(ckpt_supports)))
        
        # 填充矩阵数据
        for i, support in enumerate(support_nums):
            for j, ckpt_support in enumerate(ckpt_supports):
                values = df_filtered[(df_filtered['support_num'] == support) & 
                                   (df_filtered['ckpt_support_num'] == ckpt_support)][metric]
                if len(values) > 0:
                    heatmap_data[i,j] = values.iloc[0]
        
        # 绘制热力图
        sns.heatmap(heatmap_data, 
                   xticklabels=ckpt_supports,
                   yticklabels=support_nums,
                   annot=True, 
                   fmt='.2f',
                   cmap='Blues',
                #    vmin=80,  
                #    vmax=95,  
                   ax=ax)
        
        ax.set_title(f'Heatmap of {metric}')
        ax.set_xlabel('Checkpoint Support Number')
        ax.set_ylabel('Segmentation Support Number')
    plt.tight_layout()
    # plt.savefig(f'figures/support_heatmap_metrics_{"mean" if use else "median"}.png')
    plt.savefig(f'figures/support_heatmap_metrics_{use}.png')
    
    plt.clf()  # 关闭图形以释放内存


# # 绘制图表
# # 例如，观察ckpt_support=500, ckpt_iter=40时的变化
# # breakpoint()
# plot_metrics_by_step(df, 500, 40)

# # 观察step=0时不同ckpt_support的效果
# plot_metrics_by_ckpt_support(df, 90, 40)

# plot_metrics_by_ckpt(df)

# fixed_ckpt_iter = 50
# fixed_step = 10
# for fixed_step in [10,20,30,40,50,60,70,80,90,100]:
#     for fixed_ckpt_iter in [20,30,40,50,]:
#         print(f'fixed_step:{fixed_step},fixed_ckpt_iter:{fixed_ckpt_iter}')
#         draw_heatmap(df,fixed_ckpt_iter,fixed_step)


# draw_heatmap_average(df,use='mean')
draw_heatmap_average(df,use='max')


'''
cd /224045019/6051_final_project/GiT/
python cmd_extract.py
'''