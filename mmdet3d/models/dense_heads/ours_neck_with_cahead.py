import torch
from torch import nn
import MinkowskiEngine as ME
from mmdet.core import reduce_mean, build_assigner
from mmdet.models.builder import HEADS, build_loss
from mmcv.cnn import Scale, bias_init_with_prob
from mmdet3d.ops import furthest_point_sample

from mmdet3d.ops.pcdet_nms import pcdet_nms_gpu, pcdet_nms_normal_gpu



@HEADS.register_module()
class Fcaf3DNeckWithHead_ours(nn.Module):
    def __init__(self,
                 n_classes,
                 in_channels,
                 out_channels,
                 n_reg_outs,
                 voxel_size,
                 pts_threshold,
                 assigner,
                 yaw_parametrization='fcaf3d',
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoU3DLoss', loss_weight=1.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 noise_stages=2,
                 noise_ratio=0.8,
                 n_cross_attention=2048):
        super(Fcaf3DNeckWithHead_ours, self).__init__()
        self.voxel_size = voxel_size
        self.yaw_parametrization = yaw_parametrization
        self.assigner = build_assigner(assigner)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pts_threshold = pts_threshold
        self._init_layers(in_channels, out_channels, n_reg_outs, n_classes)
        self.stage2_reg = nn.Linear(out_channels, n_reg_outs)
        self.stage2_cls = nn.Linear(out_channels, n_classes)
        self.noise_stages = noise_stages
        self.noise_ratios = [noise_ratio*(i+1) /
                             noise_stages for i in range(noise_stages)]
        self.n_cross_attention = n_cross_attention

    @staticmethod
    def _make_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )

    @staticmethod
    def _make_up_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )

    def _init_layers(self, in_channels, out_channels, n_reg_outs, n_classes):
        # neck layers
        self.pruning = ME.MinkowskiPruning()
        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(f'up_block_{i}', self._make_up_block(in_channels[i], in_channels[i - 1]))
            self.__setattr__(f'out_block_{i}', self._make_block(in_channels[i], out_channels))

        # head layers
        self.centerness_conv = ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, dimension=3)
        self.reg_conv = ME.MinkowskiConvolution(out_channels, n_reg_outs, kernel_size=1, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(out_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.scales = nn.ModuleList([Scale(1.) for _ in range(len(in_channels))])

    def init_weights(self):
        nn.init.normal_(self.centerness_conv.kernel, std=.01)
        nn.init.normal_(self.reg_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

    def forward(self, x):
        outs = []
        outs_features = []
        inputs = x
        x = inputs[-1]
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = inputs[i] + x
                x = self._prune(x, scores)

            out = self.__getattr__(f'out_block_{i}')(x)
            out = self.forward_single(out, self.scales[i])
            scores = out[-1]
            outs_features.append(out[-2])
            outs.append(out[:-2])
        
        batch_size = len(outs[-1][0])
        # turned to scale 1 to 4, 1 is the max scale
        outs = outs[::-1]
        outs_features = outs_features[::-1]
        # inference/training
        points = []
        features = []
        sort_inds = []
        cross_features = []
        for i in range(batch_size):
            _centernesses = [out[0][i] for out in outs]
            _cls_scores = [out[2][i] for out in outs]
            _points = [out[3][i] for out in outs]
            _features = [out[i] for out in outs_features]
            _centernesses = torch.cat(_centernesses, dim=0).squeeze()
            _cls_scores = torch.cat(_cls_scores, dim=0)
            _points = torch.cat(_points, dim=0)
            _features = torch.cat(_features, dim=0)
            # select topk
            select_scores = _cls_scores.sigmoid() * _centernesses.sigmoid().unsqueeze(dim=-1)
            max_scores, _ = select_scores.max(dim=1)
            # cross_inds = torch.topk(max_scores, self.n_cross_attention)[1]
            cross_inds = furthest_point_sample(_points.unsqueeze(0), self.n_cross_attention)[0]
            # import IPython
            # IPython.embed()
            # exit()
            cross_features.append(_features[cross_inds.long()])
            
            max_k = max_scores.shape[0]
            top_k = min(max_k, 256)
            inds = torch.topk(max_scores, top_k)[1]
            _sort_inds = torch.sort(inds)[0]
            # denoising
            if self.training:
                for noise in self.noise_ratios:
                    num_queries_random = int(noise * top_k)
                    num_queries_sampling = top_k - num_queries_random
                    query_inds = torch.topk(max_scores, num_queries_sampling)[1]

                    p = torch.rand(max_scores.shape[0], )
                    p[query_inds] = 0.0

                    random_inds = torch.topk(p, num_queries_random)[1]
                    dn_inds = torch.cat([query_inds, random_inds.to(query_inds.device)])
                    _sort_dn_inds = torch.sort(dn_inds)[0]
                    _sort_inds = torch.cat([_sort_inds, _sort_dn_inds])
            features.append(_features[_sort_inds])
            points.append(_points[_sort_inds])
            sort_inds.append(_sort_inds)
        
        features = torch.stack(features, dim=0)
        cross_features = torch.stack(cross_features, dim=0)
        sort_inds = torch.stack(sort_inds, dim=0)
        points = torch.stack(points, dim=0)
        
        # return zip(*outs[::-1])
        return zip(*outs), (points, features, sort_inds, cross_features)

    def _prune(self, x, scores):
        if self.pts_threshold < 0:
            return x

        with torch.no_grad():
            coordinates = x.C.float()
            interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = interpolated_scores.new_zeros((len(interpolated_scores)), dtype=torch.bool)
            for permutation in x.decomposition_permutations:
                score = interpolated_scores[permutation]
                mask = score.new_zeros((len(score)), dtype=torch.bool)
                topk = min(len(score), self.pts_threshold)
                ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
                mask[ids] = True
                prune_mask[permutation[mask]] = True
        x = self.pruning(x, prune_mask)
        return x

    def loss(self,
             centernesses,
             bbox_preds,
             cls_scores,
             points,
             gt_bboxes,
             gt_labels,
             img_metas):
        assert len(centernesses[0]) == len(bbox_preds[0]) == len(cls_scores[0]) \
               == len(points[0]) == len(img_metas) == len(gt_bboxes) == len(gt_labels)

        loss_centerness, loss_bbox, loss_cls = [], [], []
        targets = []
        for i in range(len(img_metas)):
            img_loss_centerness, img_loss_bbox, img_loss_cls, img_targets = self._loss_single(
                centernesses=[x[i] for x in centernesses],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_scores=[x[i] for x in cls_scores],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i]
            )
            loss_centerness.append(img_loss_centerness)
            loss_bbox.append(img_loss_bbox)
            loss_cls.append(img_loss_cls)
            targets.append(img_targets)
        return dict(
            loss_centerness=torch.mean(torch.stack(loss_centerness)),
            loss_bbox=torch.mean(torch.stack(loss_bbox)),
            loss_cls=torch.mean(torch.stack(loss_cls))
        ), targets

    # per image
    def _loss_single(self,
                     centernesses,
                     bbox_preds,
                     cls_scores,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta):
        with torch.no_grad():
            centerness_targets, bbox_targets, labels = self.assigner.assign(points, gt_bboxes, gt_labels)

        centerness = torch.cat(centernesses)
        bbox_preds = torch.cat(bbox_preds)
        cls_scores = torch.cat(cls_scores)
        points = torch.cat(points)

        # skip background
        pos_inds = torch.nonzero(labels >= 0).squeeze(1)
        n_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=centerness.device)
        n_pos = max(reduce_mean(n_pos), 1.)
        loss_cls = self.loss_cls(cls_scores, labels, avg_factor=n_pos)
        pos_centerness = centerness[pos_inds]
        pos_bbox_preds = bbox_preds[pos_inds]
        pos_centerness_targets = centerness_targets[pos_inds].unsqueeze(1)
        pos_bbox_targets = bbox_targets[pos_inds]
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = points[pos_inds]
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=n_pos
            )
            loss_bbox = self.loss_bbox(
                self._bbox_pred_to_bbox(pos_points, pos_bbox_preds),
                pos_bbox_targets,
                weight=pos_centerness_targets.squeeze(1),
                avg_factor=centerness_denorm
            )
        else:
            loss_centerness = pos_centerness.sum()
            loss_bbox = pos_bbox_preds.sum()
        
        targets = (centerness_targets, bbox_targets, labels)
        return loss_centerness, loss_bbox, loss_cls, targets

    def get_bboxes(self,
                   centernesses,
                   bbox_preds,
                   cls_scores,
                   points,
                   img_metas,
                   rescale=False):
        assert len(centernesses[0]) == len(bbox_preds[0]) == len(cls_scores[0]) \
               == len(points[0]) == len(img_metas)
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                centernesses=[x[i] for x in centernesses],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_scores=[x[i] for x in cls_scores],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
            )
            results.append(result)
        return results

    # per image
    def _get_bboxes_single(self,
                           centernesses,
                           bbox_preds,
                           cls_scores,
                           points,
                           img_meta,):
        mlvl_bboxes, mlvl_scores = [], []
        for centerness, bbox_pred, cls_score, point in zip(
            centernesses, bbox_preds, cls_scores, points
        ):
            scores = cls_score.sigmoid() * centerness.sigmoid()
            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                bbox_pred = bbox_pred[ids]
                scores = scores[ids]
                point = point[ids]

            bboxes = self._bbox_pred_to_bbox(point, bbox_pred)
            mlvl_bboxes.append(bboxes)
            # print(bboxes.shape)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        # bboxes, scores, labels = self._nms(bboxes, scores, img_meta)
        # return bboxes, scores, labels
        return bboxes, scores

    # per scale
    def forward_single(self, x, scale):
        centerness = self.centerness_conv(x).features
        scores = self.cls_conv(x)
        cls_score = scores.features
        prune_scores = ME.SparseTensor(
            scores.features.max(dim=1, keepdim=True).values,
            coordinate_map_key=scores.coordinate_map_key,
            coordinate_manager=scores.coordinate_manager)
        reg_final = self.reg_conv(x).features
        reg_distance = torch.exp(scale(reg_final[:, :6]))
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_distance, reg_angle), dim=1)

        centernesses, bbox_preds, cls_scores, points = [], [], [], []
        for permutation in x.decomposition_permutations:
            centernesses.append(centerness[permutation])
            bbox_preds.append(bbox_pred[permutation])
            cls_scores.append(cls_score[permutation])

        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size
        features = x.decomposed_features

        return centernesses, bbox_preds, cls_scores, points, features, prune_scores

    def _bbox_pred_to_bbox(self, points, bbox_pred):
        if bbox_pred.shape[0] == 0:
            return bbox_pred.view(-1,7)

        x_center = points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2
        y_center = points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2
        z_center = points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2

        # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max -> x, y, z, w, l, h
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 0] + bbox_pred[:, 1],
            bbox_pred[:, 2] + bbox_pred[:, 3],
            bbox_pred[:, 4] + bbox_pred[:, 5],
        ], -1)

        if bbox_pred.shape[1] == 6:
            return base_bbox

        if self.yaw_parametrization == 'naive':
            # ..., alpha
            return torch.cat((
                base_bbox,
                bbox_pred[:, 6:7]
            ), -1)
        elif self.yaw_parametrization == 'sin-cos':
            # ..., sin(a), cos(a)
            norm = torch.pow(torch.pow(bbox_pred[:, 6:7], 2) + torch.pow(bbox_pred[:, 7:8], 2), 0.5)
            sin = bbox_pred[:, 6:7] / norm
            cos = bbox_pred[:, 7:8] / norm
            return torch.cat((
                base_bbox,
                torch.atan2(sin, cos)
            ), -1)
        else:  # self.yaw_parametrization == 'fcaf3d'
            # ..., sin(2a)ln(q), cos(2a)ln(q)
            scale = bbox_pred[:, 0] + bbox_pred[:, 1] + bbox_pred[:, 2] + bbox_pred[:, 3]
            q = torch.exp(torch.sqrt(torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
            alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
            return torch.stack((
                x_center,
                y_center,
                z_center,
                scale / (1 + q),
                scale / (1 + q) * q,
                bbox_pred[:, 5] + bbox_pred[:, 4],
                alpha
            ), dim=-1)

    def _nms(self, bboxes, scores, img_meta):
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = pcdet_nms_gpu
            else:
                class_bboxes = torch.cat((
                    class_bboxes, torch.zeros_like(class_bboxes[:, :1])), dim=1)
                nms_function = pcdet_nms_normal_gpu

            nms_ids, _ = nms_function(class_bboxes, class_scores, self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(bboxes.new_full(class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes, box_dim=box_dim, with_yaw=with_yaw, origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels
