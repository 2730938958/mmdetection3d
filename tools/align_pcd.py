import open3d as o3d
import torch
import numpy as np
import os
from pathlib import Path
import glob


def load_bin_point_cloud(file_path):
    """
    加载KITTI的.bin格式点云文件（velodyne数据）
    :param file_path: bin文件路径
    :return: open3d的PointCloud对象
    """
    # 读取bin文件（KITTI velodyne格式为float32，每4个值为x,y,z,intensity）
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    # 只保留x,y,z坐标
    points = points[:, :3].astype(np.float32)

    # 确保点云形状是 (N, 3)
    if points.ndim != 2 or points.shape[1] != 3:
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


def visualize_registered_pair(pcd_path1, pcd_path2, window_name="连续帧配准结果"):
    """
    可视化两个配准后的点云文件（区分颜色）
    :param pcd_path1: 第一帧点云路径（红色）
    :param pcd_path2: 第二帧点云路径（蓝色）
    """
    # 加载点云
    pcd1 = o3d.io.read_point_cloud(pcd_path1)
    pcd2 = o3d.io.read_point_cloud(pcd_path2)

    # 设置颜色
    pcd1.paint_uniform_color([1, 0, 0])  # 第一帧：红色
    pcd2.paint_uniform_color([0, 0, 1])  # 第二帧：蓝色

    # 可视化
    o3d.visualization.draw_geometries([pcd1, pcd2], window_name=window_name)


def save_transformation_matrix(transform_matrix, save_dir, file_name):
    """
    保存变换矩阵到指定目录
    :param transform_matrix: 4x4变换矩阵（numpy数组）
    :param save_dir: 保存目录
    :param file_name: 保存的文件名（不含后缀）
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    # 拼接完整路径（保存为npy格式，方便后续加载使用）
    save_path = os.path.join(save_dir, f"{file_name}.npy")
    # 保存矩阵
    np.save(save_path, transform_matrix)
    print(f"变换矩阵已保存: {save_path}")


def batch_register_consecutive_frames(input_dir, output_dir, matrix_save_dir, voxel_size=0.05, icp_threshold=0.02):
    """
    批量处理连续帧点云配准
    :param input_dir: 输入bin点云目录
    :param output_dir: 输出配准后点云目录
    :param matrix_save_dir: 变换矩阵保存目录
    :param voxel_size: 粗配准体素大小
    :param icp_threshold: ICP距离阈值
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取所有bin文件，按六位数命名排序
    bin_files = sorted(glob.glob(os.path.join(input_dir, "*.bin")))
    if len(bin_files) < 2:
        print("错误：输入目录下bin文件数量少于2个，无法进行连续帧配准")
        return

    print(f"共检测到 {len(bin_files)} 个bin点云文件，开始连续两两配准...")

    # 遍历连续帧对（i和i+1）
    for i in range(len(bin_files) - 1):
        # 获取当前帧和下一帧路径
        curr_bin = bin_files[i]
        next_bin = bin_files[i + 1]

        # 提取文件名（用于输出命名）
        curr_name = Path(curr_bin).stem
        next_name = Path(next_bin).stem
        # 生成变换矩阵文件名（如000001to000002）
        matrix_file_name = f"{curr_name}to{next_name}"
        print(f"\n===== 处理帧对: {curr_name} -> {next_name} =====")

        try:
            # 1. 加载点云
            print(f"加载点云: {curr_bin} 和 {next_bin}")
            source_pcd = load_bin_point_cloud(curr_bin)  # 当前帧作为源点云
            target_pcd = load_bin_point_cloud(next_bin)  # 下一帧作为目标点云
            print(f"源点云点数: {len(source_pcd.points)}, 目标点云点数: {len(target_pcd.points)}")

            # 2. 粗配准
            print("执行FPFH+RANSAC粗配准...")
            coarse_result = ransac_coarse_registration(source_pcd, target_pcd, voxel_size)
            print(f"粗配准拟合度: {coarse_result.fitness:.6f}")

            # 3. 精配准
            print("执行点到平面ICP精配准...")
            fine_result = point_to_plane_icp_fine_registration(
                source_pcd, target_pcd, coarse_result.transformation, icp_threshold
            )
            print(f"精配准拟合度: {fine_result.fitness:.6f}, 内点RMSE: {fine_result.inlier_rmse:.6f}")

            # 4. 应用变换到源点云（当前帧配准到下一帧坐标系）
            registered_source = source_pcd.transform(fine_result.transformation)

            # 5. 保存配准后的点云（保存为pcd格式，兼容Open3D可视化）
            # 保存配准后的当前帧
            curr_output_path = os.path.join(output_dir, f"{curr_name}_aligned.pcd")
            o3d.io.write_point_cloud(curr_output_path, registered_source)
            # 保存原始目标帧（方便后续可视化配对）
            next_output_path = os.path.join(output_dir, f"{next_name}_original.pcd")
            o3d.io.write_point_cloud(next_output_path, target_pcd)

            # 6. 保存精配准变换矩阵
            save_transformation_matrix(
                transform_matrix=fine_result.transformation,
                save_dir=matrix_save_dir,
                file_name=matrix_file_name
            )

            print(f"配准结果已保存:")
            print(f"  - 配准后的{curr_name}: {curr_output_path}")
            print(f"  - 原始的{next_name}: {next_output_path}")

        except Exception as e:
            print(f"处理帧对 {curr_name} -> {next_name} 失败: {e}")
            continue

    print("\n===== 批量配准完成 =====")


# 主程序
if __name__ == "__main__":
    # 配置路径
    INPUT_DIR = "/home/users/weiyan/mmdetection3d/data/kitti/training/velodyne_reduced"
    OUTPUT_DIR = "/home/users/weiyan/mmdetection3d/data/kitti/training/align_pcd"
    MATRIX_SAVE_DIR = "/home/users/weiyan/mmdetection3d/data/kitti/training/align_matrix"

    # 批量执行连续帧配准
    batch_register_consecutive_frames(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        matrix_save_dir=MATRIX_SAVE_DIR,
        voxel_size=0.05,  # 可根据点云尺度调整
        icp_threshold=0.02
    )
