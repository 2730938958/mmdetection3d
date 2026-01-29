# model settings
_base_ = './pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'
# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian']
metainfo = dict(classes=class_names)
backend_args = None

point_cloud_range = [-10.24, -10.24, -3, 10.24, 10.24, 3]
voxel_size = [0.16, 0.16, 6]
model = dict(
    # 2. 覆盖data_preprocessor：统一voxel_layer的point_cloud_range
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,  # 继承基础配置的参数
            max_voxels=(16000, 40000),  # 继承基础配置的参数
            point_cloud_range=point_cloud_range,  # 统一为自定义范围
            voxel_size=voxel_size  # 保持与基础配置一致
        )
    ),
    # 3. 覆盖voxel_encoder：同步point_cloud_range（否则柱体编码会用基础配置的范围）
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,  # 保持与基础配置一致
        point_cloud_range=point_cloud_range  # 统一为自定义范围
    ),
    # 4. 你原有的bbox_head配置（已正确设置锚点范围，无需修改）
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='AlignedAnchor3DRangeGenerator',
            ranges=[point_cloud_range],  # 已与自定义范围统一
            sizes=[[0.6, 0.8, 1.73]],  # 行人尺寸正确
            rotations=[0, 1.57],  # 行人朝向正确
            reshape_out=True)),
    # 5. 放宽正样本阈值（关键！解决正样本过少问题）
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.4,  # 降低正样本IoU阈值（原0.6，行人难达）
            neg_iou_thr=0.3,  # 降低负样本IoU阈值（原0.45，扩大正样本候选）
            min_pos_iou=0.2,  # 降低最小正样本IoU（原0.45，避免漏检）
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False,
        code_weight=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0]
    ))

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Pedestrian=5)),
    classes=class_names,
    sample_groups=dict(Pedestrian=100),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=False),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    dataset=dict(dataset=dict(pipeline=train_pipeline, metainfo=metainfo)))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',  # 确保路径正确
    metric='bbox',
    # 关键：添加激光雷达范围，与你的point_cloud_range一致
    pcd_limit_range=point_cloud_range,  # 显式传递自定义范围
    backend_args=backend_args)

test_evaluator = val_evaluator  # 测试评估器与验证评估器保持一致