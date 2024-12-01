auto_scale_lr = dict(base_batch_size=24)
backend_args = None
base_img_size = 1120
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=1000, max_keep_ckpts=1,
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
det_cfgs = dict(
    global_only_image=True,
    grid_interpolate=True,
    grid_resolution_perwin=[
        5,
        5,
    ],
    max_decoder_length=5,
    mode='detection',
    num_vocal=2322,
    samples_grids_eachwin=10)
det_test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        1120,
        1120,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_dict=dict(
            git_cfg=dict(
                global_only_image=True,
                grid_interpolate=True,
                grid_resolution_perwin=[
                    5,
                    5,
                ],
                max_decoder_length=5,
                mode='detection',
                num_vocal=2322,
                samples_grids_eachwin=10),
            head_cfg=dict(
                dec_length=5, num_bins=2240, num_classes=80, num_vocal=2322),
            task_name='detection'),
        type='AddMetaInfo'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'task_name',
            'head_cfg',
            'git_cfg',
        ),
        type='PackDetInputs'),
]
det_train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_dict=dict(
            git_cfg=dict(
                global_only_image=True,
                grid_interpolate=True,
                grid_resolution_perwin=[
                    5,
                    5,
                ],
                max_decoder_length=5,
                mode='detection',
                num_vocal=2322,
                samples_grids_eachwin=10),
            head_cfg=dict(
                arg_max_inference=True,
                dec_length=5,
                num_bins=2240,
                num_classes=80,
                num_vocal=2322),
            task_name='detection'),
        type='AddMetaInfo'),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=False,
                    scales=[
                        (
                            1120,
                            1120,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            400,
                            4200,
                        ),
                        (
                            500,
                            4200,
                        ),
                        (
                            600,
                            4200,
                        ),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type='absolute_range',
                    type='RandomCrop'),
                dict(
                    keep_ratio=False,
                    scales=[
                        (
                            1120,
                            1120,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
    dict(min_gt_bbox_wh=(
        1e-05,
        1e-05,
    ), type='FilterAnnotations'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'task_name',
            'head_cfg',
            'git_cfg',
        ),
        type='PackDetInputs'),
]
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=4000)
max_iters = 120000
model = dict(
    backbone=dict(
        arch='base',
        drop_path_rate=0.1,
        img_size=1120,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth',
            prefix='backbone.',
            type='Pretrained'),
        new_more_layers=[
            'win',
            'win',
            'win',
            'win',
            'win',
            'win',
        ],
        out_channels=0,
        out_type='featmap',
        patch_size=16,
        type='ViTGiT',
        use_abs_pos=True,
        use_checkpoints=True,
        use_rel_pos=True,
        window_size=14),
    bert_embed=dict(
        hidden_size=768, pretrain_path='./bert_embed.pt', type='bert-base'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_seg=True,
        pad_size_divisor=224,
        seg_pad_value=255,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='GeneralDataPreprocessor'),
    head_list=dict(
        detection_head=dict(
            test_cfg=dict(max_per_img=100),
            train_cfg=dict(
                assigner=dict(
                    match_costs=[
                        dict(
                            box_format='xywh', type='PointsL1Cost',
                            weight=5.0),
                    ],
                    type='HungarianAssigner')),
            type='GiTDetHead')),
    support_tasks=[
        'detection',
        'semantic_segmentation',
        'instance_segmentation',
        'caption',
        'grounding',
    ],
    tokenizer=dict(name_or_path='bert-base-uncased', type='BlipTokenizer'),
    type='GiT',
    use_checkpoints=True)
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone': dict(lr_mult=0.1),
            'backbone.layers.10': dict(lr_mult=0.7429),
            'backbone.layers.11': dict(lr_mult=0.8714),
            'backbone.layers.12': dict(lr_mult=1.0),
            'backbone.layers.13': dict(lr_mult=1.0),
            'backbone.layers.14': dict(lr_mult=1.0),
            'backbone.layers.15': dict(lr_mult=1.0),
            'backbone.layers.16': dict(lr_mult=1.0),
            'backbone.layers.17': dict(lr_mult=1.0),
            'backbone.layers.6': dict(lr_mult=0.2286),
            'backbone.layers.7': dict(lr_mult=0.3571),
            'backbone.layers.8': dict(lr_mult=0.4858),
            'backbone.layers.9': dict(lr_mult=0.6143),
            'reference_points': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1)
        })),
    type='OptimWrapper')
param_scheduler = [
    dict(
        T_max=120000,
        begin=0,
        by_epoch=False,
        end=120000,
        eta_min=2e-06,
        type='CosineAnnealingLR'),
]
pretrained = 'https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth'
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=3,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                1120,
                1120,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_dict=dict(
                    git_cfg=dict(
                        global_only_image=True,
                        grid_interpolate=True,
                        grid_resolution_perwin=[
                            5,
                            5,
                        ],
                        max_decoder_length=5,
                        mode='detection',
                        num_vocal=2322,
                        samples_grids_eachwin=10),
                    head_cfg=dict(
                        dec_length=5,
                        num_bins=2240,
                        num_classes=80,
                        num_vocal=2322),
                    task_name='detection'),
                type='AddMetaInfo'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'task_name',
                    'head_cfg',
                    'git_cfg',
                ),
                type='PackDetInputs'),
        ],
        return_classes=True,
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=3,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        1120,
        1120,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_dict=dict(
            git_cfg=dict(
                global_only_image=True,
                grid_interpolate=True,
                grid_resolution_perwin=[
                    5,
                    5,
                ],
                max_decoder_length=5,
                mode='detection',
                num_vocal=2322,
                samples_grids_eachwin=10),
            head_cfg=dict(
                dec_length=5, num_bins=2240, num_classes=80, num_vocal=2322),
            task_name='detection'),
        type='AddMetaInfo'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'task_name',
            'head_cfg',
            'git_cfg',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(
    max_iters=120000, type='IterBasedTrainLoop', val_interval=5000)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=3,
    dataset=dict(
        datasets=[
            dict(
                ann_file='annotations/instances_train2017.json',
                backend_args=None,
                data_prefix=dict(img='train2017/'),
                data_root='data/coco/',
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        meta_dict=dict(
                            git_cfg=dict(
                                global_only_image=True,
                                grid_interpolate=True,
                                grid_resolution_perwin=[
                                    5,
                                    5,
                                ],
                                max_decoder_length=5,
                                mode='detection',
                                num_vocal=2322,
                                samples_grids_eachwin=10),
                            head_cfg=dict(
                                arg_max_inference=True,
                                dec_length=5,
                                num_bins=2240,
                                num_classes=80,
                                num_vocal=2322),
                            task_name='detection'),
                        type='AddMetaInfo'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(
                        transforms=[
                            [
                                dict(
                                    keep_ratio=False,
                                    scales=[
                                        (
                                            1120,
                                            1120,
                                        ),
                                    ],
                                    type='RandomChoiceResize'),
                            ],
                            [
                                dict(
                                    keep_ratio=True,
                                    scales=[
                                        (
                                            400,
                                            4200,
                                        ),
                                        (
                                            500,
                                            4200,
                                        ),
                                        (
                                            600,
                                            4200,
                                        ),
                                    ],
                                    type='RandomChoiceResize'),
                                dict(
                                    allow_negative_crop=True,
                                    crop_size=(
                                        384,
                                        600,
                                    ),
                                    crop_type='absolute_range',
                                    type='RandomCrop'),
                                dict(
                                    keep_ratio=False,
                                    scales=[
                                        (
                                            1120,
                                            1120,
                                        ),
                                    ],
                                    type='RandomChoiceResize'),
                            ],
                        ],
                        type='RandomChoice'),
                    dict(
                        min_gt_bbox_wh=(
                            1e-05,
                            1e-05,
                        ),
                        type='FilterAnnotations'),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'scale_factor',
                            'flip',
                            'flip_direction',
                            'task_name',
                            'head_cfg',
                            'git_cfg',
                        ),
                        type='PackDetInputs'),
                ],
                return_classes=True,
                type='CocoDataset'),
        ],
        ignore_keys=[
            'reduce_zero_label',
            'label_map',
            'classes',
            'palette',
        ],
        type='ConcatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        batch_size=3,
        if_group=[
            True,
        ],
        shuffle=True,
        source_ratio=[
            1.0,
        ],
        type='GroupMultiSourceNonMixedSampler'))
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=3,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                1120,
                1120,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_dict=dict(
                    git_cfg=dict(
                        global_only_image=True,
                        grid_interpolate=True,
                        grid_resolution_perwin=[
                            5,
                            5,
                        ],
                        max_decoder_length=5,
                        mode='detection',
                        num_vocal=2322,
                        samples_grids_eachwin=10),
                    head_cfg=dict(
                        dec_length=5,
                        num_bins=2240,
                        num_classes=80,
                        num_vocal=2322),
                    task_name='detection'),
                type='AddMetaInfo'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'task_name',
                    'head_cfg',
                    'git_cfg',
                ),
                type='PackDetInputs'),
        ],
        return_classes=True,
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=3,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '.'
