# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional
import torch
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import InstanceList
from .two_stage import TwoStage3DDetector
from collections import defaultdict
import numpy as np

@MODELS.register_module()
class PointVoxelRCNN(TwoStage3DDetector):
    r"""PointVoxelRCNN detector.

    Please refer to the `PointVoxelRCNN <https://arxiv.org/abs/1912.13192>`_.

    Args:
        voxel_encoder (dict): Point voxelization encoder layer.
        middle_encoder (dict): Middle encoder layer
            of points cloud modality.
        backbone (dict): Backbone of extracting points features.
        neck (dict, optional): Neck of extracting points features.
            Defaults to None.
        rpn_head (dict, optional): Config of RPN head. Defaults to None.
        points_encoder (dict, optional): Points encoder to extract point-wise
            features. Defaults to None.
        roi_head (dict, optional): Config of ROI head. Defaults to None.
        train_cfg (dict, optional): Train config of model.
            Defaults to None.
        test_cfg (dict, optional): Train config of model.
            Defaults to None.
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`Det3DDataPreprocessor`. Defaults to None.
    """

    def __init__(self,
                 voxel_encoder: dict,
                 middle_encoder: dict,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 rpn_head: Optional[dict] = None,
                 points_encoder: Optional[dict] = None,
                 roi_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 num_loss = False) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)
        self.points_encoder = MODELS.build(points_encoder)
        self.num_loss = num_loss
        self.density_head = DensityHead(512, 1)
        self.scene_base_num_factor = nn.Parameter(torch.tensor(1.0))
        self.num_exp_factor = nn.Parameter(torch.tensor(1.0))
        self.density_criterion = nn.SmoothL1Loss(reduction='none', beta=1.0)
        self.bev_mask_placeholder = nn.Parameter(torch.randn(256))
        if self.num_loss == False:
            self.num_exp_factor.requires_grad = False
            self.scene_base_num_factor.requires_grad = False
            self.bev_mask_placeholder.requires_grad = False
            for param in self.density_head.parameters():
                param.requires_grad = False

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'voxels' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - voxels (dict[torch.Tensor]): Voxels of the batch sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C >=7.
        """
        feats_dict = self.extract_feat(batch_inputs_dict)
        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(feats_dict,
                                                     batch_data_samples)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        # extrack points feats by points_encoder
        points_feats_dict = self.extract_points_feat(batch_inputs_dict,
                                                     feats_dict,
                                                     rpn_results_list)

        results_list_3d = self.roi_head.predict(points_feats_dict,
                                                rpn_results_list,
                                                batch_data_samples)

        # connvert to Det3DDataSample
        results_list = self.add_pred_to_datasample(batch_data_samples,
                                                   results_list_3d)

        return results_list

    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        """Extract features from the input voxels.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'voxels' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - voxels (dict[torch.Tensor]): Voxels of the batch sample.

        Returns:
            dict: We typically obtain a dict of features from the backbone +
                neck, it includes:

                - spatial_feats (torch.Tensor): Spatial feats from middle
                    encoder.
                - multi_scale_3d_feats (list[torch.Tensor]): Multi scale
                    middle feats from middle encoder.
                - neck_feats (torch.Tensor): Neck feats from neck.
        """
        feats_dict = dict()
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        feats_dict['spatial_feats'], feats_dict[
            'multi_scale_3d_feats'] = self.middle_encoder(
                voxel_features, voxel_dict['coors'], batch_size)
        x = self.backbone(feats_dict['spatial_feats'])
        if self.with_neck:
            neck_feats = self.neck(x)
            feats_dict['neck_feats'] = neck_feats
        return feats_dict

    def extract_points_feat(self, batch_inputs_dict: dict, feats_dict: dict,
                            rpn_results_list: InstanceList) -> dict:
        """Extract point-wise features from the raw points and voxel features.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'voxels' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - voxels (dict[torch.Tensor]): Voxels of the batch sample.
            feats_dict (dict): Contains features from the first stage.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.

        Returns:
            dict: Contain Point-wise features, include:
                - keypoints (torch.Tensor): Sampled key points.
                - keypoint_features (torch.Tensor): Gather key points features
                    from multi input.
                - fusion_keypoint_features (torch.Tensor): Fusion
                    keypoint_features by point_feature_fusion_layer.
        """
        return self.points_encoder(batch_inputs_dict, feats_dict,
                                   rpn_results_list)

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'voxels' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - voxels (dict[torch.Tensor]): Voxels of the batch sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: A dictionary of loss components.
        """
        feats_dict = self.extract_feat(batch_inputs_dict)

        losses = dict()

        if self.num_loss:
            # Calculate num_loss using voxel information
            voxels = batch_inputs_dict['voxels']
            order = torch.tensor([0, 2, 3, 1])
            coors_batch = voxels['coors'][:, order]  # (total_pillars, 4) [batch_idx, x, y, z]
            npoints_voxel = voxels['num_points']  # (total_pillars,)
            pillar_coords, npoints_per_pillar = convert_voxels_to_pillar_num_points(coors_batch, npoints_voxel)


            num_point_label, actual_batch_size = build_num_point_label_vectorized(
                pillar_coords, npoints_per_pillar, down_factor=8, grid_size=64
            )


            voxel_dict = batch_inputs_dict['voxels']
            voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                                voxel_dict['num_points'],
                                                voxel_dict['coors'])
            batch_size = voxel_dict['coors'][-1, 0].item() + 1
            spatial_feats, multi_scale_3d_feats = self.middle_encoder(
                voxel_features, voxel_dict['coors'], batch_size)


            # print(self.bev_mask_placeholder)
            downsampled_coords = downsample_4d_voxels(pillar_coords, down_factor=8)
            mask_rate = 0.1  # Should match the value used in pretraining
            M = downsampled_coords.shape[0]
            down_mask = torch.rand(M, device=downsampled_coords.device) > mask_rate
            masked_down_coords = downsampled_coords[~down_mask]
            mask_label = build_mask_label_vectorized(masked_down_coords, actual_batch_size, grid_size=64)
            mask = mask_label.bool().repeat(1, 256, 1, 1)
            masked_spatial_feats = mask_pillar_features_vectorized(spatial_feats, masked_down_coords, self.bev_mask_placeholder)
            x = self.backbone(masked_spatial_feats)
            if self.with_neck:
                neck_feats = self.neck(x)
            density_pred = self.density_head(neck_feats[0])
            density = self.num_exp_factor * torch.exp(density_pred) + self.scene_base_num_factor

            per_pixel_loss = self.density_criterion(density, num_point_label)
            num_loss = (per_pixel_loss * mask_label).sum() / (mask_label.sum() + 1e-8)
            losses['num_loss'] = num_loss*0.05



        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                feats_dict,
                rpn_data_samples,
                proposal_cfg=proposal_cfg,
                **kwargs)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in keys:
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            # TODO: Not support currently, should have a check at Fast R-CNN
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        points_feats_dict = self.extract_points_feat(batch_inputs_dict,
                                                     feats_dict,
                                                     rpn_results_list)

        roi_losses = self.roi_head.loss(points_feats_dict, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        use_entropy_weight = True
        if use_entropy_weight:
            label_paths = [data_samples.lidar_path.replace('velodyne_reduced', 'label_2').replace('bin', 'txt') for data_samples in
             batch_data_samples]
            scores = [read_last_float_from_txt(p) for p in label_paths]
            entropy = [calculate_uncertainty(np.array(score), 0.4) for score in scores]
            entropy = np.mean(np.array(entropy))
            for k in losses.keys():
                if type(losses[k]) is torch.Tensor:
                    losses[k] = losses[k] * entropy * 2
                else:
                    losses[k][0] = losses[k][0] * entropy * 2

        return losses


def downsample_4d_voxels(voxel_coords, down_factor=2):
    """
    下采样四维体素坐标（含batch维度）
    Args:
        voxel_coords: [N, 4] 张量，格式为 [batch_idx, x, y, z]
                      x,y范围0-127，z固定，batch_idx为批次索引
        down_factor: 下采样因子（此处固定为2，128→64）
    Returns:
        unique_downsampled: [M, 4] 张量，下采样后的非空体素坐标（含batch）
    """
    # 定义每个维度的下采样因子：batch和z不变（因子=1），x和y因子=2
    factors = torch.tensor([1, down_factor, down_factor, 1], device=voxel_coords.device)
    # 对坐标进行整除下采样（仅x,y变化）
    downsampled = voxel_coords // factors
    # 对四维坐标去重（保留每个批次内的唯一下采样体素）
    unique_downsampled = torch.unique(downsampled, dim=0)
    return unique_downsampled


def build_num_point_label_vectorized(coors_batch, npoints_per_pillar, down_factor=2, grid_size=64):
    # 1. 计算所有原始体素对应的下采样坐标
    coors = coors_batch.long()  # [总数量, 4]：[b, x, y, z]
    b = coors[:, 0]
    x = coors[:, 1]
    y = coors[:, 2]

    # 下采样坐标（x'=x//2, y'=y//2）
    x_prime = x // down_factor
    y_prime = y // down_factor

    # 2. 过滤超出64×64网格的坐标（避免越界）
    valid = (x_prime < grid_size) & (y_prime < grid_size) & (x_prime >= 0) & (y_prime >= 0)
    b_valid = b[valid]
    xp_valid = x_prime[valid]
    yp_valid = y_prime[valid]
    points_valid = npoints_per_pillar[valid].float()  # 有效体素的点数

    # 3. 计算批次大小
    batch_size = coors[:, 0].max().item() + 1 if coors.numel() > 0 else 1

    # 4. 用scatter_add_向量化累加每个下采样区域的点数
    num_point_label = torch.zeros(
        (batch_size, 1, grid_size, grid_size),
        device=coors.device,
        dtype=torch.float32
    )
    # 展平索引：b * grid_size^2 + x' * grid_size + y'（确保唯一索引）
    indices = b_valid * grid_size * grid_size + xp_valid * grid_size + yp_valid
    num_point_label.view(-1).scatter_add_(0, indices, points_valid)

    return num_point_label, batch_size


def build_mask_label_vectorized(masked_down_coords, batch_size, grid_size=64):
    if masked_down_coords.numel() == 0:
        return torch.zeros((batch_size, 1, grid_size, grid_size), device=masked_down_coords.device)

    # 提取坐标并过滤越界值
    b = masked_down_coords[:, 0].long()
    x_prime = masked_down_coords[:, 1].long()
    y_prime = masked_down_coords[:, 2].long()
    valid = (x_prime < grid_size) & (y_prime < grid_size) & (b < batch_size) & (x_prime >= 0) & (y_prime >= 0)
    b_valid = b[valid]
    xp_valid = x_prime[valid]
    yp_valid = y_prime[valid]

    # 向量化标记mask区域
    mask_label = torch.zeros(
        (batch_size, 1, grid_size, grid_size),
        device=masked_down_coords.device,
        dtype=torch.float32
    )
    indices = b_valid * grid_size * grid_size + xp_valid * grid_size + yp_valid
    mask_label.view(-1)[indices] = 1.0  # 向量化赋值

    return mask_label



import torch.nn as nn
class DensityHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(
            self.conv.weight,
            mode='fan_in',
            nonlinearity='relu'
        )
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        return out


def convert_voxels_to_pillar_num_points(voxel_coords, voxel_num_points):
    # 步骤1-3：同之前（提取BEV坐标、哈希编码、合并点数）
    bev_coords = voxel_coords[:, [0, 1, 2]]  # (N, 3)
    zero_col = torch.zeros_like(bev_coords[:, :1])
    bev_coords = torch.cat([bev_coords, zero_col], dim=1)
    max_x = 512
    max_y = 512
    batch_indices = bev_coords[:, 0].long()
    x_indices = bev_coords[:, 1].long()
    y_indices = bev_coords[:, 2].long()
    hash_ids = batch_indices * (max_x * max_y) + x_indices * max_y + y_indices  # (N,)

    # 3. 合并同一pillar的点数（得到M个pillar的总点数）
    unique_hash_ids, inverse_indices = torch.unique(hash_ids, return_inverse=True)
    M = unique_hash_ids.numel()
    pillar_num_points = torch.zeros(M, dtype=voxel_num_points.dtype, device=voxel_num_points.device)
    pillar_num_points.scatter_add_(dim=0, index=inverse_indices, src=voxel_num_points)

    # 4. 提取唯一pillar的坐标（兼容低版本PyTorch，无return_index时）
    # 4.1 对hash_ids排序，得到排序后的哈希值和原始索引
    sorted_hash, sorted_indices = torch.sort(hash_ids)  # sorted_indices是原始索引
    # 4.2 找到排序后哈希值的唯一值位置（相邻重复值的第一个索引）
    # 用diff找到相邻元素不同的位置，再补充第一个元素的索引
    is_unique = torch.cat([torch.tensor([True], device=hash_ids.device), sorted_hash[1:] != sorted_hash[:-1]])
    unique_sorted_indices = sorted_indices[is_unique]  # 唯一哈希值在原始数据中的索引
    # 4.3 提取唯一pillar的坐标
    pillar_coords = bev_coords[unique_sorted_indices]  # (M, 3)

    # 验证维度匹配
    assert pillar_coords.shape[0] == M, f"pillar_coords长度错误：{pillar_coords.shape[0]} != {M}"
    return pillar_coords, pillar_num_points


def get_voxel_mask_region(mask_label, voxel_coords, down_factor=8):
    """
    通过mask_label构建原始分辨率mask，并判断哪些voxel属于mask区域
    Args:
        mask_label: (B, 1, H_down, W_down) 下采样分辨率的掩码（1表示掩码区域）
        voxel_coords: (N, 4) 体素坐标 [batch_idx, x, y, z]（原始分辨率）
        down_factor: 下采样因子（如8）
    Returns:
        original_mask: (B, 1, H_original, W_original) 原始分辨率的掩码
        voxel_in_mask: (N,) 布尔张量，True表示体素属于mask区域
    """
    B, _, H_down, W_down = mask_label.shape
    device = mask_label.device

    # 步骤1：将mask_label扩展到原始分辨率（H_original = H_down * down_factor，W_original = W_down * down_factor）
    original_mask = mask_label.repeat_interleave(down_factor, dim=2)  # 高度方向扩展 down_factor 倍
    original_mask = original_mask.repeat_interleave(down_factor, dim=3)  # 宽度方向扩展 down_factor 倍
    H_original, W_original = original_mask.shape[2], original_mask.shape[3]

    # 步骤2：提取体素的batch_idx、x、y坐标（原始分辨率）
    batch_indices = voxel_coords[:, 0].long()  # (N,) 体素所属批次
    x_coords = voxel_coords[:, 1].long()  # (N,) 原始x坐标
    y_coords = voxel_coords[:, 2].long()  # (N,) 原始y坐标

    # 过滤越界坐标（避免索引错误）
    valid = (x_coords < W_original) & (y_coords < H_original) & (x_coords >= 0) & (y_coords >= 0)
    x_coords = x_coords.clamp(0, W_original - 1)
    y_coords = y_coords.clamp(0, H_original - 1)

    # 步骤3：判断每个体素是否属于mask区域
    # 用高级索引获取每个体素在original_mask中的对应值
    voxel_in_mask = torch.zeros_like(valid, dtype=torch.bool, device=device)
    voxel_in_mask[valid] = original_mask[
        batch_indices[valid],  # 批次索引
        0,  # 通道索引（固定为0）
        y_coords[valid],  # 原始y坐标（对应mask的高度维度）
        x_coords[valid]  # 原始x坐标（对应mask的宽度维度）
    ].bool()

    return original_mask, voxel_in_mask


def read_last_float_from_txt(file_path):
    result_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # 遍历每一行
        for line_num, line in enumerate(f, 1):
            # 去除行首尾的空白字符（换行、空格等）
            stripped_line = line.strip()

            # 跳过空行
            if not stripped_line:
                continue

            # 按空格分割行内容（处理多个连续空格的情况）
            parts = stripped_line.split()

            # 检查该行是否有足够的元素
            if len(parts) < 1:
                print(f"警告: 第{line_num}行内容为空或格式异常，已跳过")
                continue
            try:
                # 取最后一个元素并转换为浮点数
                last_float = float(parts[-1])
                result_list.append(last_float)
            except ValueError:
                # 处理无法转换为浮点数的情况
                print(f"警告: 第{line_num}行最后一个元素'{parts[-1]}'不是有效的浮点数，已跳过")
    return result_list

def batch_find_original_voxels(downsampled_coords, coors_batch, down_factor=2):
    coors = coors_batch.long()  # [N,4]：[b, x, y, z]
    b = coors[:, 0]
    x = coors[:, 1]
    y = coors[:, 2]
    z = coors[:, 3]
    x_prime = x // down_factor
    y_prime = y // down_factor
    # 下采样坐标元组：(b, x', y', z) → 转为整数编码（便于哈希）
    down_hash = b * (64 * 64 * 2) + x_prime * (64 * 2) + y_prime * 2 + z  # 假设z唯一（0或1）

    # 2. 对downsampled_coords做同样的哈希编码
    ds_coors = downsampled_coords.long()  # [M,4]
    ds_b = ds_coors[:, 0]
    ds_xp = ds_coors[:, 1]
    ds_yp = ds_coors[:, 2]
    ds_z = ds_coors[:, 3]
    ds_hash = ds_b * (64 * 64 * 2) + ds_xp * (64 * 2) + ds_yp * 2 + ds_z

    # 3. 用哈希表建立 {下采样哈希值: 原始坐标索引列表}
    hash_map = defaultdict(list)
    for idx, h in enumerate(down_hash.cpu().tolist()):  # CPU哈希更快
        hash_map[h].append(idx)

    # 4. 按downsampled_coords的顺序提取原始坐标
    ori_coor_list = []
    for h in ds_hash.cpu().tolist():
        indices = hash_map.get(h, [])
        if indices:
            ori_coor_list.append(coors_batch[indices])
        else:
            ori_coor_list.append(torch.empty((0,4), device=coors_batch.device))
    return ori_coor_list

def calculate_uncertainty(ei, threshold):
    mask = ei > threshold
    total_uncertainty = np.sum(ei) / len(ei) * np.sum(mask) / len(ei)
    return total_uncertainty

def mask_pillar_features_vectorized(pillar_features, coords, placeholder):
    if coords.numel() == 0:
        return pillar_features

    b_indices = coords[:, 0]
    h_indices = coords[:, 1]
    w_indices = coords[:, 2]
    z_indices = coords[:, 3]
    pillar_features[b_indices, :, h_indices, w_indices] = placeholder

    return pillar_features