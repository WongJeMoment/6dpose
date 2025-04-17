import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

# 指定GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append('..')
from utils.dataset import getDataset
from models.tracker_eval import *

if __name__ == '__main__':
    ckpt_path = '/home/wangzhe/ICRA2025/E-3DTrack/ckpt.pth'
    data_folder = '/home/wangzhe/ICRA2025/E-3DTrack/E-3DTrack'

    # 加载模型
    model = TrackerNetEval(feature_dim=384, hgnn=True).to(device)
    model_info = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(model_info["state_dict"])
    print(f"Loaded from: {ckpt_path}")

    # 加载测试数据集
    testDataset = getDataset(data_folder=data_folder, train=False)

    with torch.no_grad():
        model.eval()
        for n, td in enumerate(testDataset):
            DL = DataLoader(td, batch_size=1)
            model.reset()
            opFolder = os.path.join('./output', td.sequence_name)
            os.makedirs(opFolder, exist_ok=True)

            for i, (ts, data, gt) in enumerate(DL):
                # 把 data 和 gt 都放到 CUDA 上
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(device)
                for k, v in gt.items():
                    if isinstance(v, torch.Tensor):
                        gt[k] = v.to(device)

                current_pos_l = data['u_centers_l']
                ref_patch = data['ref_img']
                pred = None
                pos_l = []
                disp = []
                pos_3d = []

                for unroll in range(data['ev_frame_left'].shape[1]):
                    ev_frame_l = data['ev_frame_left'][:, unroll]
                    ev_frame_r = data['ev_frame_right'][:, unroll]

                    flow_l_pred, disp_pred, pred = model(ev_frame_l, ev_frame_r, ref_patch, current_pos_l, None, pred=pred)

                    current_pos_l = current_pos_l + flow_l_pred.detach()
                    pos = td.reprojectImageTo3D_ph(disp_pred[0].cpu(), current_pos_l[0].cpu())

                    disp.append(disp_pred)
                    pos_l.append(current_pos_l.clone())
                    pos_3d.append(pos)

                pos_3d = torch.stack(pos_3d)
                np.save(os.path.join(opFolder, 'pos_3d_pred.npy'), np.array(pos_3d.cpu()))
                np.save(os.path.join(opFolder, 'pos_3d_gt.npy'), np.array(gt['track_3d'][0].transpose(1, 0).cpu()))

                print(f'Test, Sequence: [{n+1}]/[{len(testDataset)}]')
