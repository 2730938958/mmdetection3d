_base_ = './pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'
tag = 'eval'
# 数据集配置（仅保留推理相关）
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian']
metainfo = dict(classes=class_names)
backend_args = None

# 点云范围和体素大小（全局统一）
point_cloud_range = [-10.24, -10.24, -3, 10.24, 10.24, 3]
voxel_size = [0.16, 0.16, 6]

# 模型配置（重点修改推理相关部分）
model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            max_voxels=(16000, 40000),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size
        )
    ),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range
    ),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,  # 单类（行人）
        anchor_generator=dict(
            _delete_=True,
            type='AlignedAnchor3DRangeGenerator',
            ranges=[point_cloud_range],
            sizes=[[0.6, 0.8, 1.73]],  # 行人尺寸
            rotations=[0, 1.57],
            reshape_out=True)
    ),
    # 显式定义推理后处理参数
    test_cfg=dict(
        score_thr=0.3,  # 置信度阈值
        nms=dict(type='nms_3d', iou_threshold=0.1),
        max_num=100
    )
)

# 推理数据处理流程（简化，无增强）
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points'])
]

# 推理数据集加载配置
test_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))  # 验证与推理流程一致

# 评估器配置（保持与训练一致）
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    pcd_limit_range=point_cloud_range,
    backend_args=backend_args)
test_evaluator = val_evaluator