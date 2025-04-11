import os
import sys
import time
import threading
import numpy as np
import torch


sys.path.append("..")  # 根据你的项目结构调整
from utils.dataset import getDataset
from models.tracker_eval import TrackerNetEval

sys.path.append("/usr/lib/python3/dist-packages/")

from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent



CAMERA_CONFIGS = {
    "left": {"serial": "00051195", "mode": "slave"},
    "right": {"serial": "00051197", "mode": "master"}
}


def get_event_frame_generator(serial):
    device = initiate_device(path=serial)
    mv_iterator = EventsIterator.from_device(device=device)
    height, width = mv_iterator.get_size()
    frame_gen = PeriodicFrameGenerationAlgorithm(width, height, fps=300, palette=ColorPalette.CoolWarm)

    frames = []

    def on_frame_cb(ts, frame):
        frames.append(frame.copy())

    frame_gen.set_output_callback(on_frame_cb)

    def generator():
        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()
            frame_gen.process_events(evs)
            if frames:
                yield frames.pop(0)

    return generator()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 修改为你用的 GPU

    print("🚀 Loading model...")
    model = TrackerNetEval(feature_dim=384, hgnn=True)
    model.load_state_dict(torch.load('./ckpt.pth')["state_dict"])
    model = model.cuda().eval()
    print("✅ Model loaded.")

    print("📦 Loading dataset for geometry info...")
    dataset = getDataset(data_folder="./data", train=False)
    td = dataset[0]  # 只用第一个 sequence 的 geometry 工具
    ref_patch = td[0][1]['ref_img'].cuda()
    current_pos_l = td[0][1]['u_centers_l'].cuda()

    pred = None
    pos_3d = []
    pos_l = []
    disp = []

    print("📡 Initializing cameras...")
    left_stream = get_event_frame_generator(CAMERA_CONFIGS["left"]["serial"])
    right_stream = get_event_frame_generator(CAMERA_CONFIGS["right"]["serial"])

    print("🎬 Starting inference loop...")
    try:
        for idx, (ev_frame_l_np, ev_frame_r_np) in enumerate(zip(left_stream, right_stream)):
            if idx >= 100:  # 只处理前100帧，可根据需求修改
                break

            # 转换成模型输入格式
            ev_frame_l = torch.from_numpy(ev_frame_l_np).float().unsqueeze(0).unsqueeze(0).cuda() / 255.
            ev_frame_r = torch.from_numpy(ev_frame_r_np).float().unsqueeze(0).unsqueeze(0).cuda() / 255.

            with torch.no_grad():
                flow_l_pred, disp_pred, pred = model(
                    ev_frame_l, ev_frame_r, ref_patch, current_pos_l, None, pred=pred
                )
                current_pos_l += flow_l_pred.detach()

                pos = td.reprojectImageTo3D_ph(disp_pred[0].cpu(), current_pos_l[0].cpu())

                pos_3d.append(pos)
                pos_l.append(current_pos_l.clone())
                disp.append(disp_pred)

                print(f"🧠 Frame {idx}: 3D points estimated.")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")

    print("💾 Saving results...")
    os.makedirs('./output/realtime', exist_ok=True)
    np.save('./output/realtime/pos_3d_pred.npy', np.array(pos_3d))
    print("✅ Saved: pos_3d_pred.npy")


if __name__ == "__main__":
    main()
