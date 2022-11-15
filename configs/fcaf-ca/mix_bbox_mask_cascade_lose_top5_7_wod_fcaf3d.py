_base_ = ['./fcaf3d.py']

num_layers = 1
lr = 0.001 * 2 # max learning rate
optimizer = dict(
    type='AdamW', lr=lr, weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'decoder': dict(lr_mult=0.05, decay_mult=1.0),
        },
    ),
)

model = dict(
    type='PointCascadeSparse3DDetector',
    neck_with_head=dict(
        type='Fcaf3DNeckWithHead_cascade',
        n_classes=10,
        n_reg_outs=8,
        cross_sampling=False,
        n_cross_attention=1024,
        num_query=1024,
        noise_stages=0,
        assigner=dict(
            type='CascadeAssigner',
            scale=0.3,
            min_scale=0.3,
            topk_sampling=5,
            num_layers=num_layers+2)),
    stage2_head=dict(
        type='CAHeadPoint',
        update_points=True,
        update_ref=True,
        refine_query=True,
        refine_with_res=False,
        refine_with_bbox=True,
        with_centerness_mask=True,
        with_self_attn=False,
        average_refine=False,
        refine_with_updated_points=True,
        detach_dist=False,
        topk_refine=7,
        decoder=dict(
            type='TransformerDecoderLayerWithPosPoint',
            num_layers=num_layers,
            transformerlayers=dict(
                type='PointDetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1),
                feedforward_channels=1024,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm'),
            ),
            posembed=dict(
                input_channel=9,
                num_pos_feats=256,
            ),
        ),
    ),
    freeze_img_branch=True,
    test_cfg=dict(
        ensemble_stages=[num_layers+1],
        iou_thr=.25,
        score_thr=.01,
    ),
)
