import torch
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core import BaseAssigner
from mmdet3d.core.bbox.structures import rotation_3d_in_axis

@BBOX_ASSIGNERS.register_module()
class CascadeAssigner(BaseAssigner):
    def __init__(self, limit, topk, n_scales, num_layers, topk_sampling=1, scale=0.2, min_scale=0.5):
        self.limit = limit
        self.topk = topk
        self.n_scales = n_scales
        self.num_layers = num_layers
        self.scale = scale
        self.min_scale = min_scale
        self.topk_sampling = topk_sampling

    def center2targets(self, centers, gt_bboxes, scale):

        dx_min = centers[..., 0] - gt_bboxes[..., 0] + \
            gt_bboxes[..., 3] * scale
        dx_max = gt_bboxes[..., 0] + \
            gt_bboxes[..., 3] * scale - centers[..., 0]
        dy_min = centers[..., 1] - gt_bboxes[..., 1] + \
            gt_bboxes[..., 4] * scale
        dy_max = gt_bboxes[..., 1] + \
            gt_bboxes[..., 4] * scale - centers[..., 1]
        dz_min = centers[..., 2] - gt_bboxes[..., 2] + \
            gt_bboxes[..., 5] * scale
        dz_max = gt_bboxes[..., 2] + \
            gt_bboxes[..., 5] * scale - centers[..., 2]
        bbox_targets = torch.stack(
            (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max, gt_bboxes[..., 6]), dim=-1)
        return bbox_targets

    def assign(self, points, gt_bboxes, gt_labels, stage=None):
        float_max = 1e8
        # expand scales to align with points
        expanded_scales = [
            points[i].new_tensor(i).expand(len(points[i]))
            for i in range(len(points))
        ]
        points = torch.cat(points, dim=0)
        scales = torch.cat(expanded_scales, dim=0)

        # below is based on FCOSHead._get_target_single
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = gt_bboxes.volume.to(points.device)
        volumes = volumes.expand(n_points, n_boxes).contiguous()
        gt_bboxes = torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        gt_bboxes = gt_bboxes.to(points.device).expand(n_points, n_boxes, 7)
        expanded_points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        shift = torch.stack((
            expanded_points[..., 0] - gt_bboxes[..., 0],
            expanded_points[..., 1] - gt_bboxes[..., 1],
            expanded_points[..., 2] - gt_bboxes[..., 2]
        ), dim=-1).permute(1, 0, 2)
        shift = rotation_3d_in_axis(
            shift, -gt_bboxes[0, :, 6], axis=2).permute(1, 0, 2)
        centers = gt_bboxes[..., :3] + shift
        bbox_targets = self.center2targets(centers, gt_bboxes, scale=0.5)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets[...,
                                           :6].min(-1)[0] > 0  # skip angle

        # condition2: positive points per scale >= limit
        # calculate positive points per scale
        n_pos_points_per_scale = []
        for i in range(self.n_scales):
            n_pos_points_per_scale.append(
                torch.sum(inside_gt_bbox_mask[scales == i], dim=0))
        # find best scale
        n_pos_points_per_scale = torch.stack(n_pos_points_per_scale, dim=0)
        lower_limit_mask = n_pos_points_per_scale < self.limit
        lower_index = torch.argmax(lower_limit_mask.int(), dim=0) - 1
        lower_index = torch.where(lower_index < 0, 0, lower_index)
        all_upper_limit_mask = torch.all(
            torch.logical_not(lower_limit_mask), dim=0)
        best_scale = torch.where(all_upper_limit_mask,
                                 self.n_scales - 1, lower_index)
        # keep only points with best scale
        best_scale = torch.unsqueeze(best_scale, 0).expand(n_points, n_boxes)
        scales = torch.unsqueeze(scales, 1).expand(n_points, n_boxes)
        inside_best_scale_mask = best_scale == scales

        # condition3: limit topk locations per box by centerness
        centerness = compute_centerness(bbox_targets)
        centerness = torch.where(
            inside_gt_bbox_mask, centerness, torch.ones_like(centerness) * -1)
        centerness = torch.where(
            inside_best_scale_mask, centerness, torch.ones_like(centerness) * -1)
        top_centerness = torch.topk(centerness, min(
            self.topk + 1, len(centerness)), dim=0).values[-1]
        inside_top_centerness_mask = centerness > top_centerness.unsqueeze(0)

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        volumes = torch.where(inside_gt_bbox_mask, volumes,
                              torch.ones_like(volumes) * float_max)
        volumes = torch.where(inside_best_scale_mask,
                              volumes, torch.ones_like(volumes) * float_max)
        volumes = torch.where(inside_top_centerness_mask,
                              volumes, torch.ones_like(volumes) * float_max)
        min_area, min_area_inds = volumes.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels = torch.where(min_area == float_max, -1, labels)
        bbox_targets = bbox_targets[range(n_points), min_area_inds]
        centerness_targets = compute_centerness(bbox_targets)

        return centerness_targets, gt_bboxes[range(n_points), min_area_inds], labels

    def assign_query(self, points, gt_bboxes, gt_labels, stage=None,
                     with_centerness_mask=False, _sort_ind=None, _gt_noise_ind=None):
        float_max = 1e8
        points = torch.cat(points, dim=0)

        # below is based on FCOSHead._get_target_single
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        # import IPython
        # IPython.embed()
        # exit()
        raw_centers = gt_bboxes.tensor[:, :3].to(points.device)
        volumes = gt_bboxes.volume.to(points.device)
        raw_volumes = gt_bboxes.volume.to(points.device)
        volumes = volumes.expand(n_points, n_boxes).contiguous()
        raw_gt_bboxes = torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(points.device)
        gt_bboxes = raw_gt_bboxes.expand(n_points, n_boxes, 7)
        expanded_points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        shift = torch.stack((
            expanded_points[..., 0] - gt_bboxes[..., 0],
            expanded_points[..., 1] - gt_bboxes[..., 1],
            expanded_points[..., 2] - gt_bboxes[..., 2]
        ), dim=-1).permute(1, 0, 2)
        shift = rotation_3d_in_axis(
            shift, -gt_bboxes[0, :, 6], axis=2).permute(1, 0, 2)
        centers = gt_bboxes[..., :3] + shift
        bbox_targets = self.center2targets(centers, gt_bboxes, scale=0.5)
        assert not stage is None
        scale = (self.num_layers - stage - 1) * \
            self.scale / (self.num_layers - 1) + self.min_scale
        with torch.no_grad():
            bbox_mask = self.center2targets(centers, gt_bboxes, scale=scale)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_mask[...,
                                        :6].min(-1)[0] > 0  # skip angle
        inside_gt_bbox_mask_true = bbox_targets[...,
                                        :6].min(-1)[0] > 0  # skip angle

        centerness = compute_centerness(bbox_targets)
        centerness = torch.where(
            inside_gt_bbox_mask, centerness, torch.ones_like(centerness) * -1)
        top_centerness = torch.topk(centerness, min(
            self.topk + 1, len(centerness)), dim=0).values[-1]
        inside_top_centerness_mask = centerness > top_centerness.unsqueeze(0)

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        volumes = torch.where(inside_gt_bbox_mask, volumes,
                              torch.ones_like(volumes) * float_max)
        if with_centerness_mask:
            volumes = torch.where(inside_top_centerness_mask,
                                  volumes, torch.ones_like(volumes) * float_max)
        min_area, min_area_inds = volumes.min(dim=1)
        dist = torch.abs(raw_gt_bboxes[:, :3].unsqueeze(
            1) - points.unsqueeze(0))
        
        dist = torch.sqrt(
                torch.square(
                    dist[..., 0]) + torch.square(dist[..., 1]) + torch.square(dist[..., 2])
            )
        _gt_noise_ind_current = dist.argmin(1)
        _gt_noise_ind_current = torch.topk(dist, k=self.topk_sampling, largest=False, dim=-1)[1]
        # import IPython
        # IPython.embed()
        # exit()
        for i in range(n_boxes):
            if len(_gt_noise_ind):
                denoise_idxes = torch.where(_sort_ind == _gt_noise_ind[i])
                for denoise_idx in denoise_idxes[0]:
                    if inside_gt_bbox_mask_true[denoise_idx, i]:
                        min_area[denoise_idx] = raw_volumes[i]
                        min_area_inds[denoise_idx] = i
            if self.topk_sampling > 0:
                for denoise_idxv2 in _gt_noise_ind_current[i]:
                    if inside_gt_bbox_mask_true[denoise_idxv2, i]:
                        min_area[denoise_idxv2] = raw_volumes[i]
                        min_area_inds[denoise_idxv2] = i
                
        labels = gt_labels[min_area_inds]
        labels = torch.where(min_area == float_max, -1, labels)
        bbox_targets = bbox_targets[range(n_points), min_area_inds]
        centerness_targets = compute_centerness(bbox_targets)

        return centerness_targets, gt_bboxes[range(n_points), min_area_inds], labels

def compute_centerness(bbox_targets):
    x_dims = bbox_targets[..., [0, 1]]
    y_dims = bbox_targets[..., [2, 3]]
    z_dims = bbox_targets[..., [4, 5]]
    centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * \
                         y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * \
                         z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
    return torch.sqrt(centerness_targets)