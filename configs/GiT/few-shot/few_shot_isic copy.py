_base_ = ['../../_base_/seg_default_runtime.py',
'./git_base.py'
]
load_from = './universal_base.pth'

global_bin = _base_.global_bin
base_img_size = _base_.base_img_size
backend_args = None

ray_number = 20

isic_det_cfgs = dict(
    mode='detection',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    num_bins=global_bin, # TODO: double check
    num_classes=1,
    num_vocal=global_bin+1+1+1,
    total_num_vocal=global_bin+1+1+1,
    max_decoder_length=5,
    global_only_image=True)

isic_seg_cfgs = dict(
    # mode='semantic_segmentation',
    mode='instance_segmentation',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    num_vocal=base_img_size*2+1+1+1,
    total_num_vocal=base_img_size*2+1+1+1,
    ray_num=ray_number,
    max_decoder_length=1+4+2+ray_number,
    global_only_image=True)

isic_det_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            head_cfg=dict(num_classes=1,
                                                        num_vocal=1+1+global_bin+1,
                                                        num_bins=global_bin,
                                                        dec_length=5,
                                                        arg_max_inference=True),
                                            git_cfg=isic_det_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg')),]

isic_seg_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(base_img_size, base_img_size), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='instance_segmentation', 
                                            head_cfg=dict(num_classes=1,
                                                            num_vocal=(global_bin + 1) + 1 + 1,
                                                            num_bins=global_bin,
                                                            dec_length=1+4+2+ray_number,
                                                            arg_max_inference=True,
                                                            ray_num=ray_number,
                                                            use_mass_center=True),
                                            git_cfg=isic_seg_cfgs)),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'task_name', 'head_cfg', 'git_cfg')),]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=False),
    dataset=dict(type='CocoDataset',
                data_root='data/coco/',
                ann_file='annotations/instances_train2017.json',
                data_prefix=dict(img='train2017/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=10),
                pipeline=isic_det_train_pipeline,
                support_num=5*1,
                return_classes=True,
                backend_args=backend_args)
    )

max_iters=5
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=3)
test_cfg = dict(type='TestLoop')
val_cfg = dict(type='ValLoop')

param_scheduler = [
    dict(type='MultiStepLR', by_epoch=False, milestones=[max_iters], gamma=0.1)
]


val_dataloader = dict(
    batch_size=3,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/coco/',
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        return_classes=True,
        pipeline=isic_seg_test_pipeline))
test_pipeline = isic_seg_test_pipeline
test_dataloader = val_dataloader

# val_evaluator = dict(
#     # TODO: support WiderFace-Evaluation for easy, medium, hard cases
#     type='VOCMetric',
#     metric='mAP',
#     eval_mode='11points')

# val_evaluator = dict(type='CocoMetric',
#         ann_file='data/coco/' + 'annotations/instances_val2017.json',
#         metric='bbox',
#         format_only=False,
#         backend_args=backend_args)
# breakpoint()


val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mDice'],)
# val_evaluator = dict(
#     ann_file='data/coco/annotations/instances_val2017.json',
#     backend_args=None,
#     format_only=False,
#     metric=[
#         'bbox',
#         'segm',
#     ],
#     type='CocoMetric')


test_evaluator = val_evaluator


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=50, max_keep_ckpts=50),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook',draw=False,interval=50,show=False))

log_processor = dict(type='LogProcessor', window_size=4000, by_epoch=False)