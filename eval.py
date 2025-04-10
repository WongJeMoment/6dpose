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
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import ExtendedKalmanFilter as EKF
from scipy.optimize import least_squares

def get_virtual_cube(size=0.05):
    # 创建一个单位立方体（中心在原点）
    s = size / 2.0
    cube_points = np.array([
        [-s, -s, -s],
        [ s, -s, -s],
        [ s,  s, -s],
        [-s,  s, -s],
        [-s, -s,  s],
        [ s, -s,  s],
        [ s,  s,  s],
        [-s,  s,  s],
    ])
    return cube_points

def draw_cube(img, projected_points):
    cube_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # sides
    ]
    for i, j in cube_edges:
        pt1 = tuple(projected_points[i].astype(int))
        pt2 = tuple(projected_points[j].astype(int))
        cv2.line(img, pt1, pt2, (0, 255, 255), 2)
    return img


def pose_error(params, X, x, K):
    rvec = params[:3]
    tvec = params[3:].reshape(3, 1)
    R_mat, _ = cv2.Rodrigues(rvec)
    proj = (K @ (R_mat @ X.T + tvec)).T
    proj = proj[:, :2] / proj[:, 2:3]
    return (proj - x).ravel()

def estimate_pose_least_squares(X, x, K, rvec_init=None, tvec_init=None):
    if rvec_init is None:
        rvec_init = np.zeros(3)
    if tvec_init is None:
        tvec_init = np.zeros(3)

    params0 = np.hstack([rvec_init, tvec_init])
    result = least_squares(pose_error, params0, method='lm', args=(X, x, K))
    rvec_opt = result.x[:3].reshape(3, 1)
    tvec_opt = result.x[3:].reshape(3, 1)
    return rvec_opt, tvec_opt, result.cost, result.success


def draw_tracking(img, points, color=(0, 255, 0)):
    for pt in points:
        pt_int = tuple(pt.astype(int))
        # cv2.circle(img, pt_int, 2, color, -1)
    return img

def draw_tracking_with_info(img, points_2d, points_3d, color=(0, 255, 0)):
    for idx, (pt2d, pt3d) in enumerate(zip(points_2d, points_3d)):
        pt_int = tuple(pt2d.astype(int))
        # cv2.circle(img, pt_int, 2, color, -1)
        # label = f'{idx}: ({pt3d[0]:.2f}, {pt3d[1]:.2f}, {pt3d[2]:.2f})'
        # cv2.putText(img, label, (pt_int[0]+5, pt_int[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img

def draw_axes_at_center(img, rvec, tvec, K, object_points):
    center_3d = np.mean(object_points, axis=0).reshape((1, 3))
    axis = np.float32([[0,0,0], [0.05,0,0], [0,0.05,0], [0,0,0.05]])
    axis_translated = axis + center_3d
    imgpts, _ = cv2.projectPoints(axis_translated, rvec, tvec, K, None)
    origin = tuple(imgpts[0].ravel().astype(int))
    x_axis = tuple(imgpts[1].ravel().astype(int))
    y_axis = tuple(imgpts[2].ravel().astype(int))
    z_axis = tuple(imgpts[3].ravel().astype(int))
    cv2.line(img, origin, x_axis, (0, 0, 255), 2)
    cv2.line(img, origin, y_axis, (0, 255, 0), 2)
    cv2.line(img, origin, z_axis, (255, 0, 0), 2)
    return img

def fx(x, dt):
    return x

def hx(x):
    return x

def H_jacobian(x):
    return np.eye(len(x))

def create_ekf(dim=3, R_val=0.05, P_val=50.0, Q_val=0.005):
    ekf = EKF(dim_x=dim, dim_z=dim)
    ekf.x = np.zeros((dim, 1))
    ekf.F = np.eye(dim)
    ekf.H = np.eye(dim)
    ekf.R *= R_val
    ekf.P *= P_val
    ekf.Q *= Q_val
    ekf.fx = fx
    ekf.hx = hx
    return ekf

if __name__ == '__main__':
    ckpt_path = '/home/wangzhe/ICRA2025/E-3DTrack/ckpt.pth'
    data_folder = '/home/wangzhe/ICRA2025/E-3DTrack/E-3DTrack'

    model = TrackerNetEval(feature_dim=384, hgnn=True)
    model_info = torch.load(ckpt_path)
    model.load_state_dict(model_info["state_dict"])
    model = model.cuda()
    print(f"Loaded from: {ckpt_path}")

    testDataset = getDataset(data_folder=data_folder, train=False)

    window_name = 'Tracking - Left View'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    # 添加 Cube View 窗口
    cv2.namedWindow("Cube View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cube View", 400, 400)

    euler_ekf = create_ekf(3)
    tvec_ekf = create_ekf(3)

    with torch.no_grad():
        model.eval()
        for n, td in enumerate(testDataset):
            DL = DataLoader(td, 1)
            model.reset()
            opFolder = os.path.join('./output', td.sequence_name)
            os.makedirs(opFolder, exist_ok=True)

            for i, (ts, data, gt) in enumerate(DL):
                for k, v in data.items():
                    data[k] = v.cuda()
                for k, v in gt.items():
                    gt[k] = v.cuda()

                current_pos_l = data['u_centers_l']
                ref_patch = data['ref_img']
                pred = None
                pos_l, disp, pos_3d, tracked_2d = [], [], [], []
                rvec, tvec, ref_3d = None, None, None
                euler_smooth = euler_ekf.x.flatten()
                tvec_smooth = tvec_ekf.x.flatten()

                for unroll in range(data['ev_frame_left'].shape[1]):
                    ev_frame_l = data['ev_frame_left'][:, unroll]
                    ev_frame_r = data['ev_frame_right'][:, unroll]

                    flow_l_pred, disp_pred, pred = model(
                        ev_frame_l, ev_frame_r, ref_patch, current_pos_l, None, pred=pred
                    )

                    current_pos_l += flow_l_pred.detach()
                    pos = td.reprojectImageTo3D_ph(disp_pred[0].cpu(), current_pos_l[0].cpu())

                    disp.append(disp_pred)
                    pos_l.append(current_pos_l.clone())
                    pos_3d.append(pos)

                    ev_img = ev_frame_l[0].detach().cpu()
                    if ev_img.ndim == 3 and ev_img.shape[0] == 10:
                        ev_img_vis = ev_img.mean(dim=0).numpy() * 255
                        img = np.stack([ev_img_vis]*3, axis=-1).astype(np.uint8)
                    else:
                        raise ValueError(f"Unsupported ev_frame_l shape: {ev_img.shape}")
                    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    tracked_2d.append(current_pos_l[0].cpu().numpy())
                    tracked_3d = pos.cpu().numpy()
                    for p in tracked_2d:
                        img = draw_tracking(img, p)
                    img = draw_tracking_with_info(img, current_pos_l[0].cpu().numpy(), tracked_3d, color=(0, 0, 255))

                    K = np.array([[300.0, 0, 128], [0, 300.0, 128], [0, 0, 1]])
                    ref_3d_np = pos.cpu().numpy()
                    curr_2d_np = current_pos_l[0].cpu().numpy()

                    valid_points = min(len(ref_3d_np), len(curr_2d_np), 20)
                    if valid_points >= 6:
                        ref_3d_np = pos.cpu().numpy()
                        curr_2d_np = current_pos_l[0].cpu().numpy()
                        mask = np.all(np.isfinite(ref_3d_np), axis=1) & np.all(np.isfinite(curr_2d_np), axis=1)

                        ref_3d = ref_3d_np[mask]
                        curr_2d = curr_2d_np[mask]

                        if ref_3d.shape[0] >= 4:
                            # 选前4个点构建几何
                            # 获取所有有效点对
                            valid_mask = np.all(np.isfinite(ref_3d), axis=1) & np.all(np.isfinite(curr_2d), axis=1)
                            ref_3d_valid = ref_3d[valid_mask]
                            curr_2d_valid = curr_2d[valid_mask]

                            # 限制最大数量（避免太多点影响速度）
                            max_points = 20
                            if len(ref_3d_valid) > max_points:
                                ref_3d_valid = ref_3d_valid[:max_points]
                                curr_2d_valid = curr_2d_valid[:max_points]

                            X_template = ref_3d_valid
                            x_observed = curr_2d_valid

                            try:
                                rvec, tvec, cost, success = estimate_pose_least_squares(X_template, x_observed, K,
                                                                                        rvec_init=rvec.flatten() if rvec is not None else None,
                                                                                        tvec_init=tvec.flatten() if tvec is not None else None)
                            except Exception as e:
                                print("Least squares optimization failed:", e)
                                success = False

                        else:
                            success = False

                        if success:
                            # 投影误差可视化
                            # 虚拟立方体在模板坐标系下
                            # 用滤波后欧拉角+平移重建 Cube 姿态
                            R_smooth = R.from_euler('xyz', euler_smooth, degrees=True).as_matrix()
                            rvec_smooth, _ = cv2.Rodrigues(R_smooth)
                            tvec_smooth = tvec_smooth.reshape(3, 1)

                            cube_points_3d = get_virtual_cube()
                            cube_proj, _ = cv2.projectPoints(cube_points_3d, rvec_smooth, tvec_smooth, K, None)
                            cube_proj = cube_proj.reshape(-1, 2)

                            cube_img = np.ones_like(img) * 50  # 深灰底
                            if np.all(np.isfinite(cube_proj)):
                                cube_img = draw_cube(cube_img, cube_proj)
                            else:
                                cv2.putText(cube_img, "Invalid Projection", (50, 100),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                            cv2.imshow("Cube View", cube_img)

                            proj_2d, _ = cv2.projectPoints(X_template, rvec, tvec, K, None)
                            proj_2d = proj_2d.reshape(-1, 2)
                            for p1, p2 in zip(x_observed, proj_2d):
                                cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 0, 255), 1)

                            # EKF滤波欧拉角和平移
                            R_mat, _ = cv2.Rodrigues(rvec)
                            euler = R.from_matrix(R_mat).as_euler('xyz', degrees=True)

                            euler_ekf.predict()
                            euler_ekf.update(euler.reshape(-1, 1), H_jacobian, hx)
                            euler_smooth = euler_ekf.x.flatten()

                            tvec_ekf.predict()
                            tvec_ekf.update(tvec.reshape(-1, 1), H_jacobian, hx)
                            tvec_smooth = tvec_ekf.x.flatten()

                            # 画姿态坐标轴
                            img = draw_axes_at_center(img, rvec, tvec, K, X_template)
                        else:
                            cv2.putText(img, "Pose Estimation Failed", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 0, 255), 2)


                    else:
                        success = False
                    cv2.imshow(window_name, img)
                    key = cv2.waitKey(30)
                    if key == 27:
                        break

                pos_3d = torch.stack(pos_3d)
                np.save(os.path.join(opFolder, 'pos_3d_pred.npy'), np.array(pos_3d.cpu()))
                np.save(os.path.join(opFolder, 'pos_3d_gt.npy'), np.array(gt['track_3d'][0].transpose(1, 0).cpu()))

                print(f'Test, Sequence: [{n+1}]/[{len(testDataset)}]')

        cv2.destroyAllWindows()
