import torch
from mmdet.models.builder import HEADS
from mmdet3d.models.dense_heads.fcaf3d_neck_with_cahead import CAHeadCascade, CAHeadIter
from mmdet3d.models.model_utils import VoteModule
from mmdet.core import multi_apply
from mmdet3d.ops import build_sa_module


@HEADS.register_module()
class CAHeadCascadeVote(CAHeadCascade):
    def __init__(self,
                 update_points=True,
                 refine_point=False,
                 topk_refine=3,
                 vote_module_cfg=None,
                 vote_aggregation_cfg=None,
                 **kwargs
                 ):
        super(CAHeadCascadeVote, self).__init__(**kwargs)
        self.refine_point = refine_point
        self.topk_refine = topk_refine
        self.update_points = update_points
        self.gt_per_seed = vote_module_cfg['gt_per_seed']
        self.num_proposal = vote_aggregation_cfg['num_point']
        self.vote_module = VoteModule(**vote_module_cfg)
        self.vote_aggregation = build_sa_module(vote_aggregation_cfg)

    def transformer_decoder(self,
                            features,
                            points,
                            img_features,
                            img_metas,
                            ):
        decode_res_all = []
        # generate vote_points from seed_points
        vote_points, vote_features, vote_offset = self.vote_module(
            points, features)
        # import IPython
        # IPython.embed()
        # exit()
        aggregation_inputs = dict(
            points_xyz=vote_points, features=vote_features)
        vote_aggregation_ret = self.vote_aggregation(**aggregation_inputs)
        aggregated_points, aggregated_features, aggregated_indices = vote_aggregation_ret
        # update
        points = aggregated_points.detach()
        features = aggregated_features
        # get proposals
        predictions = self.conv_preds[0](features)
        decode_res = self.split_pred(predictions, points)
        decode_res_all.append(decode_res)

        # decoder stage
        query = features.permute(2, 0, 1)
        current_points, refine_feat = self._pred2points(
            decode_res, points, query)
        if self.update_points:
            points = current_points
        if self.refine_point:
            query = refine_feat.permute(1, 0, 2)

        # get inputs
        feat_flatten, mask_flatten, reference_points, spatial_shapes,\
            level_start_index, valid_ratios = self.prepare_decoder_inputs(
                points, img_features, img_metas)

        for i in range(self.num_decoder_layers):
            query_pos = torch.cat(
                [decode_res['distance'], decode_res['ref_points']],
                dim=-1).detach().clone()
            query = self.decoder[i](
                query=query,  # [N_query, BS, C_query]
                key=None,
                value=feat_flatten,  # [N_value, BS, C_value]
                query_pos=query_pos,  # [N_query, BS, C_query]
                key_padding_mask=mask_flatten,  # [BS, N_value]
                reference_points=reference_points,  # [BS, N_query, 2]
                spatial_shapes=spatial_shapes,  # [N_lvl, 2]
                level_start_index=level_start_index,  # [N_lvl]
                valid_ratios=valid_ratios,  # [BS, N_lvl, 2]
            )

            predictions = self.conv_preds[i+1](query.permute(1, 2, 0))
            decode_res = self.split_pred(predictions, points)
            decode_res_all.append(decode_res)
            current_points, refine_feat = self._pred2points(
                decode_res, points, query)
            if self.refine_point:
                query = refine_feat.permute(1, 0, 2)
            if self.update_points:
                points = current_points
            # reference_points = self.get_reference_points(points, img_metas)

        return decode_res_all


@HEADS.register_module()
class CAHeadVote(CAHeadIter):
    def __init__(self,
                 vote_module_cfg=None,
                 vote_aggregation_cfg=None,
                 **kwargs
                 ):
        super(CAHeadVote, self).__init__(**kwargs)
        self.gt_per_seed = vote_module_cfg['gt_per_seed']
        self.num_proposal = vote_aggregation_cfg['num_point']
        self.vote_module = VoteModule(**vote_module_cfg)
        self.vote_aggregation = build_sa_module(vote_aggregation_cfg)

    def transformer_decoder(self,
                            features,
                            seed_points,
                            img_features,
                            img_metas,
                            ):
        decode_res_all = []
        # generate vote_points from seed_points
        vote_points, vote_features, vote_offset = self.vote_module(
            seed_points, features)
        aggregation_inputs = dict(
            points_xyz=vote_points, features=vote_features)
        vote_aggregation_ret = self.vote_aggregation(**aggregation_inputs)
        aggregated_points, aggregated_features, aggregated_indices = vote_aggregation_ret
        vote_preds = {
            'seed_points': seed_points,
            'vote_points': vote_points,
            'aggregated_points': aggregated_points,
            'aggregated_indices': aggregated_indices
        }
        # update
        points = aggregated_points.detach()
        features = aggregated_features
        # get proposals
        predictions = self.conv_preds[0](features)
        decode_res = self.split_pred(predictions, points)
        decode_res_all.append(decode_res)

        # get inputs
        feat_flatten, mask_flatten, reference_points, spatial_shapes,\
            level_start_index, valid_ratios = self.prepare_decoder_inputs(
                points, img_features, img_metas)

        query = features.permute(2, 0, 1)
        for i in range(self.num_decoder_layers):
            query_pos = torch.cat(
                [decode_res['distance'], decode_res['ref_points']],
                dim=-1).detach().clone()
            query = self.decoder[i](
                query=query,  # [N_query, BS, C_query]
                key=None,
                value=feat_flatten,  # [N_value, BS, C_value]
                query_pos=query_pos,  # [N_query, BS, C_query]
                key_padding_mask=mask_flatten,  # [BS, N_value]
                reference_points=reference_points,  # [BS, N_query, 2]
                spatial_shapes=spatial_shapes,  # [N_lvl, 2]
                level_start_index=level_start_index,  # [N_lvl]
                valid_ratios=valid_ratios,  # [BS, N_lvl, 2]
            )

            predictions = self.conv_preds[i+1](query.permute(1, 2, 0))
            decode_res = self.split_pred(predictions, points)
            decode_res_all.append(decode_res)

        return decode_res_all, aggregated_indices, vote_preds

    def loss(self, preds_all, vote_preds, targets, gt_bboxes_3d, gt_labels_3d):
        seed_points = vote_preds.pop('seed_points')
        vote_targets, vote_target_masks = self.get_vote_targets(seed_points,
                                                                gt_bboxes_3d, gt_labels_3d,
                                                                None, None)
        # import IPython
        # IPython.embed()
        # exit()
        vote_loss = self.vote_module.get_loss(seed_points,
                                              vote_preds['vote_points'],
                                              vote_target_masks, vote_targets)
        # calculate vote loss
        losses_all = []
        for preds in preds_all:
            centerness = preds['centerness']
            bbox_pred = preds['bbox_pred']
            cls_scores = preds['cls_scores']
            bbox_preds = (centerness, bbox_pred, cls_scores)
            losses_all.append(self._loss(
                bbox_preds, targets, preds['ref_points']))

        losses = dict()
        assert self.num_fusion_layers + 1 == len(losses_all)
        for k in losses_all[0]:
            losses[k] = 0
            for i in range(self.num_fusion_layers + 1):
                losses[k] += losses_all[i][k] / (self.num_fusion_layers + 1)
        losses['vote_loss'] = vote_loss
        return losses

    def _loss(self, stage_preds, targets, points):
        loss_centerness, loss_bbox, loss_cls = [], [], []
        batch_size = points.shape[0]
        for i in range(batch_size):
            img_loss_centerness, img_loss_bbox, img_loss_cls = self._loss_single(
                centerness=stage_preds[0][i],
                bbox_preds=stage_preds[1][i],
                cls_scores=stage_preds[2][i],
                points=points[i],
                centerness_targets=targets[0][i],
                bbox_targets=targets[1][i],
                labels=targets[2][i],
            )
            loss_centerness.append(img_loss_centerness)
            loss_bbox.append(img_loss_bbox)
            loss_cls.append(img_loss_cls)
        return dict(
            stage2_loss_centerness=torch.mean(torch.stack(loss_centerness)),
            stage2_loss_bbox=torch.mean(torch.stack(loss_bbox)),
            stage2_loss_cls=torch.mean(torch.stack(loss_cls))
        )

    def get_vote_targets(self,
                         points,
                         gt_bboxes_3d,
                         gt_labels_3d,
                         pts_semantic_mask=None,
                         pts_instance_mask=None):
        # find empty example
        valid_gt_masks = list()
        gt_num = list()
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)
                valid_gt_masks.append(gt_labels_3d[index].new_zeros(1))
                gt_num.append(1)
            else:
                valid_gt_masks.append(gt_labels_3d[index].new_ones(
                    gt_labels_3d[index].shape))
                gt_num.append(gt_labels_3d[index].shape[0])

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for i in range(len(gt_labels_3d))]
            pts_instance_mask = [None for i in range(len(gt_labels_3d))]

        (vote_targets, vote_target_masks) = multi_apply(
            self.get_vote_targets_single, points, gt_bboxes_3d, gt_labels_3d,
            pts_semantic_mask, pts_instance_mask
        )

        vote_targets = torch.stack(vote_targets)
        vote_target_masks = torch.stack(vote_target_masks)

        return (vote_targets, vote_target_masks)

    def get_vote_targets_single(self,
                                points,
                                gt_bboxes_3d,
                                gt_labels_3d,
                                pts_semantic_mask=None,
                                pts_instance_mask=None):
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)

        # generate votes target
        num_points = points.shape[0]

        vote_targets = points.new_zeros([num_points, 3 * self.gt_per_seed])
        vote_target_masks = points.new_zeros([num_points],
                                             dtype=torch.long)
        vote_target_idx = points.new_zeros([num_points], dtype=torch.long)
        box_indices_all = gt_bboxes_3d.points_in_boxes(points)
        for i in range(gt_labels_3d.shape[0]):
            box_indices = box_indices_all[:, i]
            indices = torch.nonzero(
                box_indices, as_tuple=False).squeeze(-1)
            selected_points = points[indices]
            vote_target_masks[indices] = 1
            vote_targets_tmp = vote_targets[indices]
            votes = gt_bboxes_3d.gravity_center[i].unsqueeze(
                0) - selected_points[:, :3]

            for j in range(self.gt_per_seed):
                column_indices = torch.nonzero(
                    vote_target_idx[indices] == j,
                    as_tuple=False).squeeze(-1)
                vote_targets_tmp[column_indices,
                                 int(j * 3):int(j * 3 +
                                                3)] = votes[column_indices]
                if j == 0:
                    vote_targets_tmp[column_indices] = votes[
                        column_indices].repeat(1, self.gt_per_seed)

            vote_targets[indices] = vote_targets_tmp
            vote_target_idx[indices] = torch.clamp(
                vote_target_idx[indices] + 1, max=2)

        return vote_targets, vote_target_masks
