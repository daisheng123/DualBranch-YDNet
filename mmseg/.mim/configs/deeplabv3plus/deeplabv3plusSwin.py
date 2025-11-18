_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    type='EncoderDecoder',
    # ========== Swin-Tiny主干网络（无预训练） ==========
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        with_cp=False,
    ),
    # ========== DepthwiseSeparableASPP主解码头 ==========
    decode_head=dict(
        _delete_=True,
        type='DepthwiseSeparableASPPHead',
        in_channels=768,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=384,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN'),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)
    ),

    # ========== 新增辅助头（从Swin第2阶段提取特征） ==========
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=192,  # Swin第2阶段输出通道：96*2^1=192
        in_index=1,  # 使用主干第2阶段特征（索引从0开始）
        channels=256,  # 中间通道数
        num_convs=1,  # 辅助头卷积层数
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN'),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4)  # 辅助头损失权重（通常设为0.4）
    )
)

# ---------------------- 优化器与学习率调整 ----------------------
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-4,  # 增大学习率（因无预训练权重）
        betas=(0.9, 0.999),
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    type='OptimWrapper'
)

param_scheduler = dict(
    _delete_=True,
    policy='Poly',
    warmup='linear',
    warmup_iters=3000,  # 延长预热阶段
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)
