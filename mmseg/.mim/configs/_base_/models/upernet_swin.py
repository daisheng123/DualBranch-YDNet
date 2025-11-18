# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)  # 批归一化
backbone_norm_cfg = dict(type='LN', requires_grad=True) #层归一化
data_preprocessor = dict(
    type='SegDataPreProcessor', #使用的是segmentation的数据与处理器
    mean=[123.675, 116.28, 103.53],  #rgb图像的均值和标准差
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0, #图像填充直用于补齐图像大小
    seg_pad_val=255)  #用于标签分割的填充
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,   #每一个patch的特征维度96
        patch_size=4,    #每个图像块大小是4*4，输入的图像首先被切分为4*4的图像块，然后每个图像块将被映射到96维的向量
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
