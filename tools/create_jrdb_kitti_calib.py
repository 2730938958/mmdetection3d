import os
import shutil

# 配置参数（请根据实际情况修改）
target_dir = "/home/users/weiyan/mmdetection3d/data/kitti/testing/calib"  # 目标文件夹路径
source_file = "/home/users/weiyan/mmdetection3d/data/kitti/testing/calib/calib.txt"  # 源文件路径
start_num = 0  # 起始编号（000000）
end_num = 27660  # 结束编号（027892）

# 确保目标文件夹存在
os.makedirs(target_dir, exist_ok=True)

# 检查源文件是否存在
if not os.path.isfile(source_file):
    raise FileNotFoundError(f"源文件不存在：{source_file}")

# 循环创建文件并复制内容
for num in range(start_num, end_num + 1):
    # 生成6位数字文件名（补零）
    filename = f"{num:06d}.txt"
    # 目标文件完整路径
    target_path = os.path.join(target_dir, filename)
    # 复制源文件内容到目标文件
    shutil.copy2(source_file, target_path)  # copy2会保留源文件的元数据

print(f"已完成！共创建 {end_num - start_num + 1} 个文件")