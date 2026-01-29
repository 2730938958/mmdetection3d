from mmdet3d.apis import init_model, inference_detector
from mmdet3d.structures import LiDARInstance3DBoxes, CameraInstance3DBoxes, Box3DMode
import numpy as np
import torch
import os

# ===================== 全局配置 =====================
# 模型配置和权重路径
CONFIG_FILE = 'configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-num.py'
CHECKPOINT_FILE = '/home/users/weiyan/mmdetection3d/work_dirs/pv_rcnn_8xb2-80e_kitti-3d-num/pse_v1.pth'
# KITTI数据集根目录（需根据实际路径修改）
KITTI_ROOT = '/home/users/weiyan/mmdetection3d/data/kitti/training'
# 子目录路径
BIN_DIR = os.path.join(KITTI_ROOT, 'velodyne')  # bin点云文件目录
CALIB_DIR = os.path.join(KITTI_ROOT, 'calib')   # calib标定文件目录
# 遍历的文件数量（前1000个）
MAX_FILE_NUM = 1000
# 模型类别映射（需与训练配置的CLASSES一致）
KITTI_CLASSES = ['Pedestrian']

# ===================== 工具函数 =====================
def normalize_angle(angle):
    """将角度（弧度）归一化到[-π, π]区间"""
    if isinstance(angle, torch.Tensor):
        angle = torch.remainder(angle + np.pi, 2 * np.pi) - np.pi
    elif isinstance(angle, (np.ndarray, float, int)):
        angle = np.remainder(angle + np.pi, 2 * np.pi) - np.pi
    return angle

def parse_kitti_annotations(kitti_annotations):
    """解析KITTI标注，返回相机坐标系下的3D框参数和类别"""
    cam_bboxes = []  # 存储[x, y, z, l, w, h, ry]
    labels = []      # 存储类别名称
    for line in kitti_annotations:
        parts = line.split()
        cls = parts[0]
        # KITTI格式：h(8), w(9), l(10), x(11), y(12), z(13), ry(14)
        h = float(parts[8])
        w = float(parts[9])
        l = float(parts[10])
        x = float(parts[11])
        y = float(parts[12])
        z = float(parts[13])
        ry = float(parts[14])
        cam_bboxes.append([x, y, z, l, w, h, ry])
        labels.append(cls)
    return np.array(cam_bboxes), labels

def read_kitti_calib(calib_path):
    """读取KITTI的calib文件，返回lidar2cam矩阵（4x4）"""
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            key, value = line.split(':', 1)
            calib[key] = np.array([float(x) for x in value.split()])
    # Tr_velo_to_cam是3x4矩阵，补全为4x4齐次矩阵
    Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
    lidar2cam = np.eye(4)
    lidar2cam[:3, :4] = Tr_velo_to_cam
    return lidar2cam

def generate_kitti_annotations2(cam_bboxes, labels, scores):
    """根据检测结果生成KITTI格式的标注行"""
    kitti_annotations = []
    for i in range(len(cam_bboxes)):
        cls = labels[i]
        score = scores[i]
        v1, v2, v3, v4, v5, v6, v7 = cam_bboxes[i]
        # 填充无2D信息的默认值
        truncated = 0
        occluded = 0
        alpha = -2.0
        bbox_left = bbox_top = bbox_right = bbox_bottom = 0
        # 拼接KITTI格式行（最后添加置信度）
        line = (f"{cls} {truncated} {occluded} {alpha:.16f} "
                f"{bbox_left:} {bbox_top} {bbox_right} {bbox_bottom} "
                f"{v1:.16f} {v2:.16f} {v3:.16f} {v4:.16f} {v5:.16f} {v6:.16f} {v7:.16f} {score:.4f}")
        kitti_annotations.append(line)
    return kitti_annotations

def generate_kitti_annotations(cam_bboxes, labels, scores):
    """根据检测结果生成KITTI格式的标注行"""
    kitti_annotations = []
    for i in range(len(cam_bboxes)):
        cls = labels[i]
        score = scores[i]
        x, y, z, l, w, h, ry = cam_bboxes[i]
        # 填充无2D信息的默认值
        truncated = 0.0
        occluded = 0
        alpha = -2.0
        bbox_left = bbox_top = bbox_right = bbox_bottom = 0.0
        # 拼接KITTI格式行（最后添加置信度）
        line = (f"{cls} {truncated} {occluded} {alpha:.16f} "
                f"{bbox_left} {bbox_top} {bbox_right} {bbox_bottom} "
                f"{h:.16f} {w:.16f} {l:.16f} {x:.16f} {y:.16f} {z:.16f} {ry:.16f} {score:.4f}")
        kitti_annotations.append(line)
    return kitti_annotations

# ===================== 主处理函数 =====================
def process_single_bin(idx):
    """处理单个bin文件（输入为索引，生成六位数文件名）"""
    # 生成六位数文件名（000000 ~ 000999）
    file_id = f"{idx:06d}"
    bin_path = os.path.join(BIN_DIR, f"{file_id}.bin")
    calib_path = os.path.join(CALIB_DIR, f"{file_id}.txt")

    # 检查文件是否存在
    if not os.path.exists(bin_path):
        print(f"【{file_id}】点云文件不存在，跳过")
        return None
    if not os.path.exists(calib_path):
        print(f"【{file_id}】标定文件不存在，跳过")
        return None

    try:
        # 1. 执行3D检测
        result = inference_detector(model, bin_path)
        pred_instances = result[0].pred_instances_3d
        if len(pred_instances) == 0:
            print(f"【{file_id}】无检测结果，跳过")
            return None

        lidar_bboxes = pred_instances.bboxes_3d  # LiDAR坐标系3D框
        scores = pred_instances.scores_3d        # 置信度
        labels = pred_instances.labels_3d        # 类别索引

        # 2. 读取标定文件，获取lidar2cam矩阵
        lidar2cam = read_kitti_calib(calib_path)
        cam2lidar = np.linalg.inv(lidar2cam)     # 相机到LiDAR的逆矩阵

        # 3. LiDAR框转相机坐标系
        cam_bboxes = lidar_bboxes.convert_to(
            dst=Box3DMode.CAM,
            rt_mat=lidar2cam
        )

        # 4. 生成KITTI格式标注并解析回相机框
        kitti_labels = [KITTI_CLASSES[i] for i in labels.cpu().numpy()]
        bboxes_tensor = cam_bboxes.tensor.cpu().numpy()
        kitti_bboxes = bboxes_tensor[:, :7]
        kitti_annot_lines = generate_kitti_annotations(kitti_bboxes, kitti_labels, scores.cpu().numpy())
        cam_bboxes_np, _ = parse_kitti_annotations(kitti_annot_lines)

        # 5. 相机框转回LiDAR坐标系
        cam_bboxes_obj = CameraInstance3DBoxes(cam_bboxes_np)
        lidar_bboxes_converted = cam_bboxes_obj.convert_to(
            dst=Box3DMode.LIDAR,
            rt_mat=cam2lidar
        )

        # 6. 角度归一化并计算差值
        lidar_tensor = lidar_bboxes_converted.tensor[0]  # 转换后的LiDAR框
        pred_tensor = pred_instances.bboxes_3d.tensor[0] # 原始检测的LiDAR框
        lidar_tensor[-1] = normalize_angle(lidar_tensor[-1])
        pred_tensor[-1] = normalize_angle(pred_tensor[-1].cpu())
        diff = lidar_tensor - pred_tensor.cpu()
        new_col = [4,5,3,0,1,2,6]
        cam_bboxes_np = cam_bboxes_np[:, new_col]
        final_kitti_annot_lines = generate_kitti_annotations2(torch.tensor(cam_bboxes_np), kitti_labels, scores.cpu().numpy())
        print(f"【{file_id}】转换后与原始检测框的差值：\n{diff}")
        return {
            'file_id': file_id,
            'diff': diff,
            'lidar_tensor': lidar_tensor,
            'pred_tensor': pred_tensor
        }

    except Exception as e:
        print(f"【{file_id}】处理失败，错误：{str(e)}")
        return None

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 加载模型（只需加载一次，避免重复耗时）
    print("正在加载3D检测模型...")
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE)
    print("模型加载完成，开始遍历文件...\n")

    # 存储所有有效结果
    all_results = []
    # 遍历前1000个六位数文件（000000 ~ 000999）
    for idx in range(MAX_FILE_NUM):
        res = process_single_bin(idx)
        if res is not None:
            all_results.append(res)

    # 打印统计信息
    print(f"\n遍历完成！共处理{len(all_results)}个有效文件，跳过{MAX_FILE_NUM - len(all_results)}个文件")