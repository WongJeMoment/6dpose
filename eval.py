import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
sys.path.append('..')
from utils.dataset import getDataset
from models.tracker_eval import *
from torch.utils.data import DataLoader
import numpy as np
import torch
import cv2
import threading
import time
sys.path.append("/usr/lib/python3/dist-packages/")

from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent

def setup_camera(serial, cam_mode):
    # 打印相机信息
    print(f"\n🚀 Starting camera {serial} in mode: {cam_mode.upper()}")
    # 连接设备
    try:
        device = initiate_device(path=serial)
    except Exception as e:
        print(f"❌ Could not initiate device {serial}: {e}")
        return
    # 打印设备信息
    try:
        print(f"✅ Connected to device with serial: {device.get_serial()}")
    except:
        print("⚠️  Warning: Unable to get serial number.")

    sync_iface = device.get_i_camera_synchronization()
    if not sync_iface:
        print("❌ Device does not support synchronization interface.")
        return
    # 设置为 master / slave 模式
    try:
        if cam_mode == "master":
            sync_iface.set_mode_master()
            print("✅ Set to MASTER mode.")
        else:
            sync_iface.set_mode_slave()
            print("✅ Set to SLAVE mode.")
    except Exception as e:
        print(f"❌ Failed to set {cam_mode} mode: {e}")
        return

    # 创建显示窗口，显示事件图像
    mv_iterator = EventsIterator.from_device(device=device)
    height, width = mv_iterator.get_size()
    title = f"Metavision - {cam_mode.upper()} ({serial})"
    # 从设备读取事件流。
    with MTWindow(title=title, width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:
        # 设置键盘回调函数
        def keyboard_cb(key, scancode, action, mods):
            if key in [UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q]:
                window.set_close_flag()
        window.set_keyboard_callback(keyboard_cb)
        # 创建事件帧生成器
        frame_gen = PeriodicFrameGenerationAlgorithm(width, height, fps=300, palette=ColorPalette.CoolWarm)
        # 设置帧输出回调函数
        def on_frame_cb(ts, frame):
            window.show_async(frame)

        frame_gen.set_output_callback(on_frame_cb)

        # 事件处理主循环
        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()
            # 把当前时间段的事件丢进 frame_gen 聚合成帧。
            frame_gen.process_events(evs)
            if window.should_close():
                break


def draw_tracking(img, points, color=(0, 255, 0)):
    """ 在图像上画出点轨迹 """
    for pt in points:
        pt_int = tuple(pt.astype(int))
        cv2.circle(img, pt_int, 2, color, -1)
    return img

if __name__ == '__main__':
    ckpt_path = '/home/wangzhe/ICRA2025/E-3DTrack/ckpt.pth'
    data_folder = '/home/wangzhe/ICRA2025/E-3DTrack/E-3DTrack'

    # 加载 TrackerNetEval 模型（指定使用 HGNN，高维特征维度384）
    model = TrackerNetEval(feature_dim=384, hgnn=True)
    model_info = torch.load(ckpt_path)
    model.load_state_dict(model_info["state_dict"])
    model = model.cuda()
    # 加载测试数据集
    testDataset = getDataset(data_folder=data_folder, train=False)

    # ✅ 初始化可调节大小的窗口
    window_name = 'Tracking - Left View'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    # ✅ 逐帧处理测试数据集
    with torch.no_grad():
        # 设置模型为评估模式
        model.eval()
        # 遍历每个测试序列
        for n, td in enumerate(testDataset):
            # 读取数据
            DL = DataLoader(td, 1)
            # 重置模型状态
            model.reset()
            # 创建输出文件夹
            opFolder = os.path.join('./output', td.sequence_name)
            # 确保输出文件夹存在
            os.makedirs(opFolder, exist_ok=True)
            # 逐帧处理
            for i, (ts, data, gt) in enumerate(DL):
                # 将数据移动到 GPU
                for k, v in data.items():
                    data[k] = v.cuda()
                # 将 GT 数据移动到 GPU
                for k, v in gt.items():
                    gt[k] = v.cuda()

                # 读取数据
                current_pos_l = data['u_centers_l'] # 初始跟踪点
                ref_patch = data['ref_img']         # 参考图像 patch
                pred = None
                pos_l = []                          # 保存2D位置
                disp = []                           # 保存视差
                pos_3d = []                         # 保存3D位置
                tracked_2d = []                     # 保存轨迹
                threads = []

                # 配置事件相机
                lc = threading.Thread(target=setup_camera, args=("00051195", "slave"))
                lc.start()
                threads.append(lc)
                time.sleep(1)  # 小延迟以防初始化冲突
                rc = threading.Thread(target=setup_camera, args=("00051197", "master"))
                rc.start()
                threads.append(rc)

                for unroll in range(data['ev_frame_left'].shape[1]):
                    # 读取当前帧的事件数据
                    ev_frame_l = threading.Thread(target=setup_camera, args=("00051195", "slave"))
                    ev_frame_l.start()
                    threads.append(ev_frame_l)
                    time.sleep(1)  # 小延迟以防初始化冲突
                    ev_frame_r  = threading.Thread(target=setup_camera, args=("00051197", "master"))
                    ev_frame_r.start()
                    threads.append(ev_frame_r)
                    # ev_frame_l = data['ev_frame_left'][:, unroll]   # 左事件相机
                    # ev_frame_r = data['ev_frame_right'][:, unroll]  # 右事件相机
                    # 读取当前帧的 GT 数据
                    flow_l_pred, disp_pred, pred = model(
                        ev_frame_l, ev_frame_r, ref_patch, current_pos_l, None, pred=pred
                    )
                    current_pos_l += flow_l_pred.detach()           # 更新当前跟踪点
                    pos = td.reprojectImageTo3D_ph(disp_pred[0].cpu(), current_pos_l[0].cpu())  # 3D位置
                    # 计算当前跟踪点的3D位置
                    disp.append(disp_pred)
                    # 计算当前跟踪点的视差
                    pos_l.append(current_pos_l.clone())
                    # 保存当前跟踪点的2D位置
                    pos_3d.append(pos)

                    # ✅ 用事件帧可视化（多通道取平均）
                    ev_img = ev_frame_l[0].detach().cpu()  # shape: [10, H, W]
                    # 将事件帧转换为图像
                    if ev_img.ndim == 3 and ev_img.shape[0] == 10:
                        ev_img_vis = ev_img.mean(dim=0).numpy() * 255           # shape: [H, W]
                        img = np.stack([ev_img_vis]*3, axis=-1).astype(np.uint8)# shape: [H, W, 3]
                    else:
                        raise ValueError(f"Unsupported ev_frame_l shape: {ev_img.shape}")
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)                 # OpenCV BGR格式

                    # ✅ 绘制轨迹
                    tracked_2d.append(current_pos_l[0].cpu().numpy())
                    # 绘制当前跟踪点
                    for p in tracked_2d:
                        # 绘制轨迹
                        img = draw_tracking(img, p)
                    # 绘制当前跟踪点
                    img = draw_tracking(img, current_pos_l[0].cpu().numpy(), color=(0, 0, 255))
                    # 绘制参考图像
                    cv2.imshow(window_name, img)
                    key = cv2.waitKey(30)
                    if key == 27:  # ESC 退出
                        break

                # 逐帧处理
                # for unroll in range(data['ev_frame_left'].shape[1]):
                #     # 读取当前帧的事件数据
                #     ev_frame_l = data['ev_frame_left'][:, unroll]   # 左事件相机
                #     ev_frame_r = data['ev_frame_right'][:, unroll]  # 右事件相机
                #     # 读取当前帧的 GT 数据
                #     flow_l_pred, disp_pred, pred = model(
                #         ev_frame_l, ev_frame_r, ref_patch, current_pos_l, None, pred=pred
                #     )
                #     current_pos_l += flow_l_pred.detach()           # 更新当前跟踪点
                #     pos = td.reprojectImageTo3D_ph(disp_pred[0].cpu(), current_pos_l[0].cpu())  # 3D位置
                #     # 计算当前跟踪点的3D位置
                #     disp.append(disp_pred)
                #     # 计算当前跟踪点的视差
                #     pos_l.append(current_pos_l.clone())
                #     # 保存当前跟踪点的2D位置
                #     pos_3d.append(pos)
                #
                #     # ✅ 用事件帧可视化（多通道取平均）
                #     ev_img = ev_frame_l[0].detach().cpu()  # shape: [10, H, W]
                #     # 将事件帧转换为图像
                #     if ev_img.ndim == 3 and ev_img.shape[0] == 10:
                #         ev_img_vis = ev_img.mean(dim=0).numpy() * 255           # shape: [H, W]
                #         img = np.stack([ev_img_vis]*3, axis=-1).astype(np.uint8)# shape: [H, W, 3]
                #     else:
                #         raise ValueError(f"Unsupported ev_frame_l shape: {ev_img.shape}")
                #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)                 # OpenCV BGR格式
                #
                #     # ✅ 绘制轨迹
                #     tracked_2d.append(current_pos_l[0].cpu().numpy())
                #     # 绘制当前跟踪点
                #     for p in tracked_2d:
                #         # 绘制轨迹
                #         img = draw_tracking(img, p)
                #     # 绘制当前跟踪点
                #     img = draw_tracking(img, current_pos_l[0].cpu().numpy(), color=(0, 0, 255))
                #     # 绘制参考图像
                #     cv2.imshow(window_name, img)
                #     key = cv2.waitKey(30)
                #     if key == 27:  # ESC 退出
                #         break

                pos_3d = torch.stack(pos_3d)
                np.save(os.path.join(opFolder, 'pos_3d_pred.npy'), np.array(pos_3d.cpu()))
                np.save(os.path.join(opFolder, 'pos_3d_gt.npy'), np.array(gt['track_3d'][0].transpose(1, 0).cpu()))

                print(f'Test, Sequence: [{n}]/[{len(testDataset)}]')

        cv2.destroyAllWindows()