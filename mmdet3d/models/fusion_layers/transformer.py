from mmcv.runner.base_module import ModuleList
from mmcv.cnn.bricks.transformer import build_attention, build_feedforward_network, build_norm_layer
from mmcv import ConfigDict
from mmcv.cnn import Linear, build_activation_layer
from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.runner.base_module import BaseModule, Sequential
import copy
import warnings
import torch
import torch.nn as nn

from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import build_transformer_layer, BaseTransformerLayer

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

except ImportError:
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, cfg):
        super().__init__()
        input_channel = cfg['input_channel']
        num_pos_feats = cfg['num_pos_feats']
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


@TRANSFORMER_LAYER.register_module()
class TransformerDecoderLayerWithPos(nn.Module):
    def __init__(self, *args, transformerlayers=None, posembed=None, **kwargs):
        super().__init__()
        self.layer = build_transformer_layer(transformerlayers)
        self.posembed = PositionEmbeddingLearned(posembed)

    def init_weights(self):
        """Initialize the weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

    def forward(self,
                query,
                query_pos,
                *args,
                reference_points=None,
                valid_ratios=None,
                **kwargs):
        if reference_points.shape[-1] == 4:
            reference_points_input = reference_points[:, :, None] * \
                torch.cat([valid_ratios, valid_ratios], -1)[:, None]
        else:
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * \
                valid_ratios[:, None]

        query_pos_embed = self.posembed(query_pos)
        query_pos_embed = query_pos_embed.permute(2, 0, 1)

        output = self.layer(
            query,
            *args,
            query_pos=query_pos_embed,
            reference_points=reference_points_input,
            **kwargs)

        return output


@TRANSFORMER_LAYER.register_module()
class TransformerDecoderLayerWithPosV2(nn.Module):
    def __init__(self, *args, transformerlayers=None, posembed=None, **kwargs):
        super().__init__()
        self.layer = build_transformer_layer(transformerlayers)
        self.posembed = PositionEmbeddingLearned(posembed)

    def init_weights(self):
        """Initialize the weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

    def forward(self,
                query,
                query_pos,
                *args,
                reference_points=None,
                valid_ratios=None,
                **kwargs):
        if reference_points.shape[-1] == 4:
            reference_points_input = reference_points[:, :, None] * \
                torch.cat([valid_ratios, valid_ratios], -1)[:, None]
        else:
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * \
                valid_ratios[:, None]
        query_pos_embed = self.posembed(query_pos[:, :, :2])
        query_pos_embed = query_pos_embed.permute(2, 0, 1)

        output = self.layer(
            query,
            *args,
            query_pos=query_pos_embed,
            reference_points=reference_points_input,
            **kwargs)

        return output


@TRANSFORMER_LAYER.register_module()
class TransformerDecoderLayerWithoutPos(nn.Module):
    def __init__(self, *args, transformerlayers=None, **kwargs):
        super().__init__()
        self.layer = build_transformer_layer(transformerlayers)

    def init_weights(self):
        """Initialize the weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                **kwargs):
        if reference_points.shape[-1] == 4:
            reference_points_input = reference_points[:, :, None] * \
                torch.cat([valid_ratios, valid_ratios], -1)[:, None]
        else:
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * \
                valid_ratios[:, None]

        output = self.layer(
            query,
            *args,
            reference_points=reference_points_input,
            **kwargs)

        return output


@TRANSFORMER_LAYER.register_module()
class MaskDetrTransformerDecoderLayer(BaseTransformerLayer):
    
    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

@TRANSFORMER_LAYER.register_module()
class PIDetrTransformerDecoderLayer(BaseTransformerLayer):

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(PIDetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                point_value=None,
                query_pos=None,
                key_pos=None,
                point_key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                f'attn_masks {len(attn_masks)} must be equal ' \
                f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                point_key = point_value
                query = self.attentions[attn_index](
                    query,
                    point_key,
                    point_value,
                    identity if self.pre_norm else None,
                    query_pos=None,
                    key_pos=point_key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query



@FEEDFORWARD_NETWORK.register_module()
class C_FFN(BaseModule):

    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 **kwargs):
        super(C_FFN, self).__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        in_channels = embed_dims * 2
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


@TRANSFORMER_LAYER.register_module()
class ConditionalPIDetrTransformerDecoderLayer(BaseTransformerLayer):

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(ConditionalPIDetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        # assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                point_value=None,
                query_pos=None,
                key_pos=None,
                point_key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        base_identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                f'attn_masks {len(attn_masks)} must be equal ' \
                f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                point_key = point_value
                query = self.attentions[attn_index](
                    query,
                    point_key,
                    point_value,
                    identity if self.pre_norm else None,
                    query_pos=None,
                    key_pos=point_key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
                if norm_index == 1:
                    point_query = query

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    torch.cat([query, point_query], -1), base_identity)
                ffn_index += 1

        return query
