import os
import numpy as np
def get_pcd_file_paths(root_dir):
    """
    获取指定根目录下所有子文件夹中pcd目录的文件路径列表
    :param root_dir: 根目录路径
    :return: 所有pcd文件的绝对路径列表
    """
    pcd_file_paths = []
    # 遍历根目录下的所有子文件夹、子文件
    for parent_dir, sub_dirs, files in os.walk(root_dir):
        # 仅处理子文件夹（排除根目录本身，若需包含可移除该判断）
        if parent_dir == root_dir:
            continue
        # 拼接当前子文件夹下的pcd目录路径
        pcd_dir = os.path.join(parent_dir, "pcd")
        # 判断pcd目录是否存在
        if os.path.isdir(pcd_dir):
            # 遍历pcd目录下的所有文件
            for file_name in os.listdir(pcd_dir):
                file_path = os.path.join(pcd_dir, file_name)
                # 仅保留文件（排除子目录，若需包含可移除该判断）
                if os.path.isfile(file_path):
                    pcd_file_paths.append(file_path)
    return pcd_file_paths

# 示例调用
if __name__ == "__main__":
    # 替换为你的根目录路径（绝对路径/相对路径均可）
    root_directory = "/home/users/weiyan/go2nus"
    pcd_files = get_pcd_file_paths(root_directory)
    index = 0
    destination = '/home/users/weiyan/go2nus_kitti/pointcloud'
    for path in pcd_files:
        points = np.load(path)['arr_0']
        points[:, -1] = 1.0
        new_path = os.path.join(destination,  f"{index:06d}.bin")
        index += 1
        points.tofile(new_path)