import os
from PIL import Image  # 用于读取和保存图片
import shutil

def get_img_file_paths(root_dir):
    """
    获取指定根目录下所有子文件夹中 imgs 目录的图片路径列表
    :param root_dir: 根目录路径
    :return: 所有有效图片的绝对路径列表
    """
    img_file_paths = []
    # 支持的图片格式（可根据需要增减）
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']

    for parent_dir, sub_dirs, files in os.walk(root_dir):
        # 仅处理子文件夹（排除根目录本身，如需包含可删除该判断）
        if parent_dir == root_dir:
            continue
        # 拼接当前子文件夹下的 imgs 目录路径
        imgs_dir = os.path.join(parent_dir, "imgs")
        # 判断 imgs 目录是否存在
        if os.path.isdir(imgs_dir):
            for file_name in os.listdir(imgs_dir):
                file_path = os.path.join(imgs_dir, file_name)
                # 仅保留文件 + 符合格式的图片
                if os.path.isfile(file_path):
                    if os.path.splitext(file_name)[1].lower() in valid_extensions:
                        img_file_paths.append(file_path)
    return img_file_paths


# 示例调用
if __name__ == "__main__":
    # 1. 配置路径（根据实际情况修改）
    root_directory = "/home/users/weiyan/go2nus"  # 原始数据根目录（包含子文件夹+imgs目录）
    img_destination = "/home/users/weiyan/go2nus_kitti/training/image_2"  # 图片保存目录

    # 2. 创建目标目录（如果不存在）
    os.makedirs(img_destination, exist_ok=True)

    # 3. 获取所有 imgs 目录下的图片路径
    img_files = get_img_file_paths(root_directory)
    if not img_files:
        print("未找到任何图片文件！")
        exit()

    # 4. 按 6位索引 保存图片（000000.png, 000001.png...）
    for index, img_path in enumerate(img_files):
        try:
            # 获取原始文件的扩展名（如 .jpg、.png）
            original_ext = os.path.splitext(img_path)[1].lower()
            # 新文件名：6位索引 + 原始扩展名（保持格式不变）
            new_file_name = f"{index:06d}{original_ext}"
            # 拼接目标保存路径
            img_save_path = os.path.join(img_destination, new_file_name)
            # 复制文件（copy2 会保留文件元数据，比 copy 更完整）
            shutil.copy2(img_path, img_save_path)
            print(f"已复制：{img_path} -> {img_save_path}")
        except Exception as e:
            print(f"复制失败 {img_path}：{str(e)}")

    print(f"\n处理完成！共复制 {len(img_files)} 张图片，保存目录：{img_destination}")