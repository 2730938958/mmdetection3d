from mmdet3d.apis import init_model, inference_detector
import torch
import numpy as np
save_dir = '/home/users/weiyan/temp/vis'

npz_file = '/home/users/weiyan/temp/vis/input/92.npz'
prefix = '/home/users/weiyan/temp/vis/input'
filename = npz_file.split('/')[-1].split('.')[0]
points = np.load(npz_file)['arr_0']
points[:, -1] = 1.0
points.tofile(f'{prefix}/{filename}.bin')

config_file = 'configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-num.py'
# checkpoint_file = '/home/users/weiyan/mmdetection3d/work_dirs/pv_rcnn_8xb2-80e_kitti-3d-num/pse_v1.pth'
# checkpoint_file = '/home/users/weiyan/mmdetection3d/work_dirs/pv_rcnn_8xb2-80e_kitti-3d-num/pse_v2.pth'
checkpoint_file = '/home/users/weiyan/mmdetection3d/work_dirs/pv_rcnn_8xb2-80e_kitti-3d-num/epoch_1.pth'
pcd_file = f'{prefix}/{filename}.bin'

# pcd_file = '/home/users/weiyan/mmdetection3d/data/kitti/training/velodyne/000000.bin'
# pcd_file = '/home/users/weiyan/temp/pc.bin'

model = init_model(config_file, checkpoint_file, device='cuda:0')
cfg = model.cfg

result = inference_detector(model, pcd_file)
score = result[0].pred_instances_3d.scores_3d
threshold = 0.4
filtered_x = torch.where(score > threshold)
l = len(filtered_x[0])
torch.save(result[0].pred_instances_3d.bboxes_3d.tensor[:l, :], f'{save_dir}/boxes.pt')
torch.save(result[1]['inputs']['points'], f'{save_dir}/pc.pt')