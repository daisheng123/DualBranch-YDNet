_base_ = [
    './swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        depths=[2, 2, 18, 2]),
    decode_head=dict(in_channels=96, num_classes=2, align_corners=False,
                     channels=512,
                     dilations=[
                         1,
                         6,
                         12,
                         18,
                     ],
                     dropout_ratio=0.1,
                     in_index=0,
                     loss_decode=dict(
                         loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
                     norm_cfg=dict(requires_grad=True, type='BN'),
                     type='ASPPHead'),
    auxiliary_head=dict(in_channels=384, num_classes=2))
