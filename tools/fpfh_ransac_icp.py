import open3d as o3d
import torch
import numpy as np


def load_pt_point_cloud(file_path):
    """
    加载.pt格式的点云文件
    :param file_path: 点云文件路径
    :return: open3d的PointCloud对象
    """
    # 加载.pt文件（PyTorch张量）
    pt_data = torch.load(file_path).cpu()[:, :3]

    # 转换为numpy数组，并确保是float32类型
    if isinstance(pt_data, torch.Tensor):
        points = pt_data.numpy().astype(np.float32)
    else:
        points = np.array(pt_data).astype(np.float32)

    # 确保点云形状是 (N, 3)
    if points.ndim == 2 and points.shape[1] == 3:
        pass
    elif points.ndim == 3 and points.shape[2] == 3:
        points = points.reshape(-1, 3)
    else:
        raise ValueError(f"点云数据形状不正确，应为(N, 3)，实际为{points.shape}")

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


def compute_fpfh_feature(pcd, voxel_size):
    """
    计算FPFH特征（用于粗配准）
    :param pcd: 输入点云
    :param voxel_size: 下采样体素大小
    :return: 下采样点云、FPFH特征
    """
    # 下采样（加速特征计算）
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 估计法向量（FPFH特征计算必需）
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    # 计算FPFH特征
    radius_fpfh = voxel_size * 5
    fpfh_feature = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_fpfh, max_nn=100)
    )

    return pcd_down, fpfh_feature


def ransac_coarse_registration(source_pcd, target_pcd, voxel_size=0.05):
    """
    FPFH+RANSAC粗配准（解决大初始偏差，为精配准提供优质初始位姿）
    :param source_pcd: 源点云
    :param target_pcd: 目标点云
    :param voxel_size: 下采样体素大小
    :return: 粗配准结果
    """
    # 计算FPFH特征
    source_down, source_fpfh = compute_fpfh_feature(source_pcd, voxel_size)
    target_down, target_fpfh = compute_fpfh_feature(target_pcd, voxel_size)

    # RANSAC配准参数设置
    distance_threshold = voxel_size * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,  # 双向匹配过滤错误对应
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,  # 每次随机采样3个点计算变换
        checkers=[
            # 边缘长度检查（过滤不合理匹配）
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            # 距离检查（过滤远距离匹配）
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )

    return result_ransac


def point_to_plane_icp_fine_registration(source_pcd, target_pcd, init_transform, threshold=0.02):
    """
    点到平面ICP精配准（精度最高的ICP变种）
    :param source_pcd: 源点云
    :param target_pcd: 目标点云
    :param init_transform: 粗配准提供的初始变换矩阵
    :param threshold: 对应点对距离阈值
    :return: 精配准结果
    """
    # 估计法向量（点到平面ICP必需）
    source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    # 点到平面ICP（精度远高于原始点到点ICP）
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=threshold,
        init=init_transform,  # 使用粗配准结果作为初始变换
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),  # 点到平面优化
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=2000
        )
    )

    return reg_p2l


def visualize_point_clouds(pcd1, pcd2, window_name="高精度配准结果"):
    """
    可视化两个点云（分别用红色和蓝色显示）
    :param pcd1: 第一个点云（配准后的源点云，红色）
    :param pcd2: 第二个点云（目标点云，蓝色）
    """
    # 为点云设置不同颜色以便区分
    pcd1.paint_uniform_color([1, 0, 0])  # 红色 - 配准后的0.pt
    pcd2.paint_uniform_color([0, 0, 1])  # 蓝色 - 原始的1.pt

    # 创建可视化窗口并显示
    o3d.visualization.draw_geometries([pcd1, pcd2], window_name=window_name)


# 主程序
if __name__ == "__main__":
    # 1. 加载点云
    print("正在加载点云文件...")
    try:
        source_pcd = load_pt_point_cloud("0.pt")  # 源点云（需要配准的点云）
        target_pcd = load_pt_point_cloud("1.pt")  # 目标点云（参考点云）
        print(f"源点云点数: {len(source_pcd.points)}, 目标点云点数: {len(target_pcd.points)}")
    except Exception as e:
        print(f"加载点云失败: {e}")
        exit(1)

    # 可选：下采样（点云数量>10万时建议启用，平衡速度和精度）
    # voxel_size = 0.01
    # source_pcd = source_pcd.voxel_down_sample(voxel_size=voxel_size)
    # target_pcd = target_pcd.voxel_down_sample(voxel_size=voxel_size)

    # 2. 第一步：FPFH+RANSAC粗配准
    print("正在执行FPFH+RANSAC粗配准...")
    try:
        # 自适应体素大小（根据点云尺度调整，默认0.05适合米级点云）
        voxel_size = 0.05 if len(source_pcd.points) > 10000 else 0.01
        coarse_result = ransac_coarse_registration(source_pcd, target_pcd, voxel_size)
        print(f"粗配准拟合度: {coarse_result.fitness:.6f}")
        print("粗配准变换矩阵:")
        print(coarse_result.transformation)
    except Exception as e:
        print(f"粗配准失败: {e}")
        exit(1)

    # 3. 第二步：点到平面ICP精配准
    print("正在执行点到平面ICP精配准...")
    try:
        fine_result = point_to_plane_icp_fine_registration(
            source_pcd, target_pcd, coarse_result.transformation
        )
        # 输出精配准信息
        print(f"精配准拟合度（fitness）: {fine_result.fitness:.6f} (越接近1越好)")
        print(f"精配准内点RMSE: {fine_result.inlier_rmse:.6f} (越小越好)")
        print("最终变换矩阵:")
        print(fine_result.transformation)
    except Exception as e:
        print(f"精配准失败: {e}")
        exit(1)

    # 4. 应用最终变换矩阵到源点云
    registered_source_pcd = source_pcd.transform(fine_result.transformation)

    # 5. 可视化配准结果
    print("正在可视化配准结果...")
    visualize_point_clouds(registered_source_pcd, target_pcd)

    # 可选：保存配准后的点云
    # o3d.io.write_point_cloud("registered_0_high_precision.pcd", registered_source_pcd)
    # torch.save(torch.from_numpy(np.asarray(registered_source_pcd.points)), "registered_0_high_precision.pt")