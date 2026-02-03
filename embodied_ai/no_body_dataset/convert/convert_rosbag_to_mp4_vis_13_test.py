#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import rosbag
from cv_bridge import CvBridge
import imageio
import cv2
try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy.spatial.transform import Rotation as R
    _HAS_PLOT = True
except Exception:
    _HAS_PLOT = False


def to_sec_ros_time(t) -> float:
    # rosbag 的 t 是 rospy.Time 风格
    return float(t.to_sec()) if hasattr(t, "to_sec") else float(t)


def ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Unexpected image shape: {img.shape}, expect (H,W,3)")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def decode_compressed_image_to_rgb(msg) -> np.ndarray:
    # sensor_msgs/CompressedImage: msg.data is bytes
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode returned None (bad compressed image data?)")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return ensure_uint8_rgb(rgb)

def decode_depth_image_to_u16(msg) -> np.ndarray:
    # sensor_msgs/Image: keep raw uint16 depth to preserve precision
    arr = np.frombuffer(msg.data, dtype=np.uint16)
    depth = arr.reshape(msg.height, msg.width)
    return depth

def decode_image_msg_to_rgb(msg, bridge: CvBridge) -> np.ndarray:
    if hasattr(msg, "format") and "compressed" in msg._type.lower():
        return decode_compressed_image_to_rgb(msg)
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
    return ensure_uint8_rgb(cv_img)


def get_valid_timestamp(msg, bag_time_sec: float) -> float:
    header_ts = 0.0
    if hasattr(msg, "header"):
        try:
            header_ts = float(msg.header.stamp.to_sec())
        except Exception:
            header_ts = 0.0
    if header_ts > 1.5e9:
        return header_ts
    return float(bag_time_sec)


def estimate_fps(ts: List[float], default: float = 30.0) -> float:
    if len(ts) < 3:
        return default
    d = np.diff(np.array(ts, dtype=np.float64))
    d = d[d > 0]
    if d.size == 0:
        return default
    fps = float(1.0 / np.median(d))
    if not np.isfinite(fps) or fps <= 0:
        return default
    return float(np.clip(fps, 1.0, 120.0))


def nearest_idx(q_ts: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
    i = np.searchsorted(ref_ts, q_ts, side="left")
    i = np.clip(i, 0, len(ref_ts) - 1)
    j = np.clip(i - 1, 0, len(ref_ts) - 1)
    pick_j = np.abs(q_ts - ref_ts[j]) <= np.abs(q_ts - ref_ts[i])
    return np.where(pick_j, j, i)


def pose7_from_msg(msg) -> List[float]:
    # xv_sdk/PoseStampedConfidence: msg.poseMsg is geometry_msgs/PoseStamped
    ps = msg.poseMsg
    p = ps.pose.position
    q = ps.pose.orientation
    return [float(p.x), float(p.y), float(p.z), float(q.x), float(q.y), float(q.z), float(q.w)]


def camera_info_to_dict(msg) -> Dict[str, Any]:
    return {
        "width": int(msg.width),
        "height": int(msg.height),
        "distortion_model": str(msg.distortion_model),
        "D": [float(x) for x in msg.D],
        "K": [float(x) for x in msg.K],
        "R": [float(x) for x in msg.R],
        "P": [float(x) for x in msg.P],
        "binning_x": int(getattr(msg, "binning_x", 0)),
        "binning_y": int(getattr(msg, "binning_y", 0)),
        "roi": {
            "x_offset": int(getattr(msg.roi, "x_offset", 0)) if hasattr(msg, "roi") else 0,
            "y_offset": int(getattr(msg.roi, "y_offset", 0)) if hasattr(msg, "roi") else 0,
            "height": int(getattr(msg.roi, "height", 0)) if hasattr(msg, "roi") else 0,
            "width": int(getattr(msg.roi, "width", 0)) if hasattr(msg, "roi") else 0,
            "do_rectify": bool(getattr(msg.roi, "do_rectify", False)) if hasattr(msg, "roi") else False,
        },
    }


def tf_static_to_list(msg) -> List[Dict[str, Any]]:
    # tf2_msgs/TFMessage: transforms: list[geometry_msgs/TransformStamped]
    out = []
    for tr in msg.transforms:
        out.append({
            "parent": str(tr.header.frame_id),
            "child": str(tr.child_frame_id),
            "translation": {
                "x": float(tr.transform.translation.x),
                "y": float(tr.transform.translation.y),
                "z": float(tr.transform.translation.z),
            },
            "rotation": {
                "x": float(tr.transform.rotation.x),
                "y": float(tr.transform.rotation.y),
                "z": float(tr.transform.rotation.z),
                "w": float(tr.transform.rotation.w),
            },
        })
    return out


def quaternion_to_euler(quat):
    r = R.from_quat(quat)
    return r.as_euler("xyz")


def plot_states(pose_data, clamp_data, output_path: Path) -> None:
    if not _HAS_PLOT:
        print("[WARN] matplotlib/scipy not available; skip state plot.")
        return
    T = len(pose_data)
    steps = np.arange(T)

    pos = pose_data[:, :3] if T > 0 else np.zeros((0, 3))
    if T > 0:
        euler = np.array([quaternion_to_euler(q) for q in pose_data[:, 3:7]])
        for i in range(3):
            euler[:, i] = np.unwrap(euler[:, i])
    else:
        euler = np.zeros((T, 3))

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    pos_labels = ["X (m)", "Y (m)", "Z (m)"]
    rot_labels = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]

    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(steps, pos[:, i], "b-", linewidth=1.5)
        ax.set_title(f"Position {pos_labels[i]}")
        ax.set_xlabel("Frame")
        ax.grid(True, alpha=0.3)

    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(steps, euler[:, i], "g-", linewidth=1.5)
        ax.set_title(f"Rotation {rot_labels[i]}")
        ax.set_xlabel("Frame")
        ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    ax.plot(steps, clamp_data, "r-", linewidth=1.5, label="Clamp")
    ax.set_title("Clamp / Gripper State")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Trajectory States ({T} frames)", fontsize=14)
    plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved state plot: {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_idx", default="00000")
    # 可选：只处理某些 serial（不传就自动从 bag 里找）
    ap.add_argument("--xv_serials", nargs="*", default=None)
    args = ap.parse_args()

    bag_path = Path(args.bag)


    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root


    idx = str(args.data_idx)
    prefix = f"idx{idx}"

    # ====== 你要导出的“视频 topic”定义 ======
    # realsense color compressed
    rs_topics = {
    "rs_cam1_color": "/camera1/color/image_raw/compressed",
    "rs_cam2_color": "/camera2/color/image_raw/compressed",
    "rs_cam3_color": "/camera3/color/image_raw/compressed",
    # "rs_cam2_infra1": "/camera2/infra1/image_rect_raw/compressed",
    # "rs_cam2_infra2": "/camera2/infra2/image_rect_raw/compressed",
    }
    rs_depth_topics = {
    "rs_cam1_depth": "/camera1/aligned_depth_to_color/image_raw",
    "rs_cam2_depth": "/camera2/aligned_depth_to_color/image_raw",
    "rs_cam3_depth": "/camera3/aligned_depth_to_color/image_raw",
    }

    # xv topics (serial-dependent)
    # 我们会自动检测 bag 里有哪些 xv serial
    # 并为每个 serial 导出一个 mp4（color_camera/image）
    bridge = CvBridge()

    # ====== 先扫一遍拿 serial 列表（如果没指定） ======
    xv_serials: List[str] = []
    if args.xv_serials:
        xv_serials = list(args.xv_serials)
    else:
        with rosbag.Bag(str(bag_path), "r") as rb:
            for topic, _, _ in rb.read_messages():
                if topic.startswith("/xv_sdk/"):
                    # /xv_sdk/<serial>/...
                    parts = topic.split("/")
                    if len(parts) >= 4:
                        xv_serials.append(parts[2])
        xv_serials = sorted(list(set(xv_serials)))

    # 如果 bag 里没有 xv，也没关系
    xv_img_topics = {f"xv_{s}_color": f"/xv_sdk/{s}/color_camera/image" for s in xv_serials}
    xv_pose_topics = {f"xv_{s}_pose": f"/xv_sdk/{s}/slam/pose" for s in xv_serials}
    xv_clamp_topics = {f"xv_{s}_clamp": f"/xv_sdk/{s}/clamp/Data" for s in xv_serials}
    xv_caminfo_topics = {f"xv_{s}_caminfo": f"/xv_sdk/{s}/color_camera/camera_info" for s in xv_serials}

    # ====== 收集图像与非图像数据 ======
    images: Dict[str, List[Tuple[float, np.ndarray]]] = {name: [] for name in rs_topics.keys()}
    for name in xv_img_topics.keys():
        images[name] = []
    depth_frames: Dict[str, List[Tuple[float, np.ndarray]]] = {name: [] for name in rs_depth_topics.keys()}
    ref_images: List[Tuple[float, np.ndarray]] = []  # (t, rgb) for alignment base
    # poses: List[Tuple[float, List[float]]] = []
    # clamps: List[Tuple[float, float]] = []
    clamps_by_serial: Dict[str, List[Tuple[float, float]]] = {s: [] for s in xv_serials}
    poses_by_serial: Dict[str, List[Tuple[float, List[float]]]] = {s: [] for s in xv_serials}
    imu_topics = ["/camera1/imu", "/camera2/imu", "/camera3/imu"]
    imus: Dict[str, List[Tuple[float, Any]]] = {t: [] for t in imu_topics}

    # camera_info / tf_static (ignored in this alignment-only output)
    tf_static_topic = "/tf_static"

    # ====== 读 bag：边读边写视频、边收集 json ======
    # video topics only; depth is saved as 16-bit PNG sequence
    all_video_topics = set(rs_topics.values()) | set(xv_img_topics.values())
    # include depth topics for PNG export
    all_topics = set(all_video_topics) \
        | set(rs_depth_topics.values()) \
        | set(imu_topics) \
        | set(xv_pose_topics.values()) \
        | set(xv_clamp_topics.values()) \
        | set(xv_caminfo_topics.values()) \
        | {tf_static_topic}

    # visualization config (no input changes): ref is first xv serial; second is fixed serial if present
    if "250801DR48FP25002355" in xv_serials:
        ref_serial = "250801DR48FP25002355"
    else:
        ref_serial = xv_serials[0] if len(xv_serials) > 0 else None
    second_serial = "250801DR48FP25002606"
    if ref_serial == second_serial and len(xv_serials) > 1:
        second_serial = xv_serials[1]
    ref_topic = f"/xv_sdk/{ref_serial}/color_camera/image" if ref_serial else None
    ref_pose_topic = f"/xv_sdk/{ref_serial}/slam/pose" if ref_serial else None
    ref_clamp_topic = f"/xv_sdk/{ref_serial}/clamp/Data" if ref_serial else None
    second_topic = f"/xv_sdk/{second_serial}/color_camera/image" if second_serial else None

    with rosbag.Bag(str(bag_path), "r") as rb:
        for topic, msg, t in rb.read_messages(topics=list(all_topics)):
            # Align behavior with convert_ros_data_to_mp4.py: use bag time for all streams
            ts = to_sec_ros_time(t)

            # ---------- 视频 ----------
            if topic in rs_topics.values():
                # 找到对应 name
                name = [k for k, v in rs_topics.items() if v == topic][0]
                try:
                    rgb = decode_compressed_image_to_rgb(msg)
                    images[name].append((ts, rgb))
                except Exception as e:
                    # 遇到坏帧就跳过，避免整个任务失败
                    print(f"[WARN] skip frame topic={topic}: {e}")
        
            elif topic in rs_depth_topics.values():
                name = [k for k, v in rs_depth_topics.items() if v == topic][0]
                try:
                    depth = decode_depth_image_to_u16(msg)
                    depth_frames[name].append((ts, depth))
                except Exception as e:
                    print(f"[WARN] skip frame topic={topic}: {e}")

            elif topic in xv_img_topics.values():
                name = [k for k, v in xv_img_topics.items() if v == topic][0]
                try:
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                    rgb = ensure_uint8_rgb(cv_img)
                    images[name].append((ts, rgb))
                    if topic == ref_topic:
                        ref_images.append((ts, rgb))
                except Exception as e:
                    print(f"[WARN] skip frame topic={topic}: {e}")

            # ---------- JSON 数据 ----------
            elif topic in imu_topics:
                # sensor_msgs/Imu
                imus[topic].append((ts, msg))

            # elif topic == ref_pose_topic:
            #     poses.append((ts, pose7_from_msg(msg)))

            # elif topic == ref_clamp_topic:
            #     # xv_sdk/Clamp: msg.data
            #     clamps.append((ts, float(msg.data)))

            #     # also collect for per-serial clamp alignment
            #     if topic in xv_clamp_topics.values():
            #         serial = [
            #             s for s, tpc in xv_clamp_topics.items() if tpc == topic
            #         ][0].replace("xv_", "").replace("_clamp", "")
            #         clamps_by_serial[serial].append((ts, float(msg.data)))
            elif topic in xv_pose_topics.values():
                serial = [
                    s for s, tpc in xv_pose_topics.items() if tpc == topic
                ][0].replace("xv_", "").replace("_pose", "")
                poses_by_serial[serial].append((ts, pose7_from_msg(msg)))
                
            elif topic in xv_clamp_topics.values():
                # collect clamp for each serial
                serial = [
                    s for s, tpc in xv_clamp_topics.items() if tpc == topic
                ][0].replace("xv_", "").replace("_clamp", "")
                clamps_by_serial[serial].append((ts, float(msg.data)))

            elif topic in xv_caminfo_topics.values():
                pass
            elif topic == tf_static_topic:
                pass

    # ====== 对齐（基于 ref_topic） ======
    if not ref_topic or len(ref_images) == 0:
        raise RuntimeError("No reference XV images found for alignment.")

    ref_images.sort(key=lambda x: x[0])
    img_ts = np.array([x[0] for x in ref_images], dtype=np.float64)

    # align pose per serial (for left/right output)
    aligned_pose_by_serial: Dict[str, np.ndarray] = {}
    pose_err_by_serial: Dict[str, np.ndarray] = {}
    for s, items in poses_by_serial.items():
        if len(items) == 0:
            aligned_pose_by_serial[s] = None
            pose_err_by_serial[s] = None
            continue
        items.sort(key=lambda x: x[0])
        s_ts = np.array([t for t, _ in items], dtype=np.float64)
        s_vals = np.stack([v for _, v in items], axis=0)
        nn_s = nearest_idx(img_ts, s_ts)
        aligned_pose_by_serial[s] = s_vals[nn_s]
        pose_err_by_serial[s] = np.abs(img_ts - s_ts[nn_s])

    # align clamp per serial (for left/right output)
    aligned_clamp_by_serial: Dict[str, np.ndarray] = {}
    clamp_err_by_serial: Dict[str, np.ndarray] = {}
    for s, items in clamps_by_serial.items():
        if len(items) == 0:
            aligned_clamp_by_serial[s] = None
            clamp_err_by_serial[s] = None
            continue
        items.sort(key=lambda x: x[0])
        s_ts = np.array([t for t, _ in items], dtype=np.float64)
        s_vals = np.array([v for _, v in items], dtype=np.float32)
        nn_s = nearest_idx(img_ts, s_ts)
        aligned_clamp_by_serial[s] = s_vals[nn_s]
        clamp_err_by_serial[s] = np.abs(img_ts - s_ts[nn_s])

    fps = 30.0

    # assign left/right by fixed serial preference
    preferred_left = "250801DR48FP25002355"
    left_serial = preferred_left if preferred_left in xv_serials else (xv_serials[0] if len(xv_serials) > 0 else None)
    right_serial = None
    for s in xv_serials:
        if s != left_serial:
            right_serial = s
            break

    if left_serial is None or aligned_pose_by_serial.get(left_serial) is None:
        raise RuntimeError(f"No pose messages found for left serial: {left_serial}")
    if aligned_clamp_by_serial.get(left_serial) is None:
        raise RuntimeError(f"No clamp messages found for left serial: {left_serial}")

    plot_pose = aligned_pose_by_serial[left_serial]
    plot_clamp = aligned_clamp_by_serial[left_serial]

    # align IMU to ref image timestamps
    imu_aligned: Dict[str, Any] = {}
    imu_err: Dict[str, Any] = {}
    for topic, items in imus.items():
        if len(items) == 0:
            imu_aligned[topic] = None
            imu_err[topic] = None
            continue
        items.sort(key=lambda x: x[0])
        imu_ts = np.array([t for t, _ in items], dtype=np.float64)
        nn_imu = nearest_idx(img_ts, imu_ts)
        aligned_list = []
        err_list = []
        for i in range(len(img_ts)):
            msg = items[nn_imu[i]][1]
            aligned_list.append(
                {
                    "orientation": {
                        "x": float(msg.orientation.x),
                        "y": float(msg.orientation.y),
                        "z": float(msg.orientation.z),
                        "w": float(msg.orientation.w),
                    },
                    "angular_velocity": {
                        "x": float(msg.angular_velocity.x),
                        "y": float(msg.angular_velocity.y),
                        "z": float(msg.angular_velocity.z),
                    },
                    "linear_acceleration": {
                        "x": float(msg.linear_acceleration.x),
                        "y": float(msg.linear_acceleration.y),
                        "z": float(msg.linear_acceleration.z),
                    },
                }
            )
            err_list.append(float(abs(img_ts[i] - imu_ts[nn_imu[i]])))
        imu_aligned[topic] = aligned_list
        imu_err[topic] = err_list

    # ====== 写 mp4（全部对齐到 ref image timestamps） ======
    video_meta = {}
    for name, frames in images.items():
        if len(frames) == 0:
            print(f"[WARN] skip empty stream: {name}")
            continue
        frames.sort(key=lambda x: x[0])
        ts_list = np.array([t for t, _ in frames], dtype=np.float64)
        nn_idx = nearest_idx(img_ts, ts_list)
        aligned_frames = [frames[i][1] for i in nn_idx]
        mp4 = out_dir / f"{prefix}_{name}.mp4"
        with imageio.get_writer(str(mp4), fps=fps) as w:
            for f in aligned_frames:
                w.append_data(f)
        topic = rs_topics.get(name, xv_img_topics.get(name, rs_depth_topics.get(name, None)))
        if topic is not None:
            video_meta[name] = {
                "file": mp4.name,
                "topic": topic,
                "frame_count": len(aligned_frames),
            }

    # ====== 写 depth PNG（16-bit，保留精度） ======
    depth_meta = {}
    for name, frames in depth_frames.items():
        if len(frames) == 0:
            print(f"[WARN] skip empty depth stream: {name}")
            continue
        frames.sort(key=lambda x: x[0])
        ts_list = np.array([t for t, _ in frames], dtype=np.float64)
        nn_idx = nearest_idx(img_ts, ts_list)
        aligned_depth = [frames[i][1] for i in nn_idx]

        depth_dir = out_dir / f"{prefix}_{name}_png"
        depth_dir.mkdir(parents=True, exist_ok=True)
        for i, d in enumerate(aligned_depth):
            cv2.imwrite(str(depth_dir / f"{i:06d}.png"), d)

        depth_meta[name] = {
            "dir": depth_dir.name,
            "topic": rs_depth_topics.get(name),
            "frame_count": len(aligned_depth),
            "format": "png16",
        }

    # ====== JSON 输出（与 convert_ros_data_to_mp4.py 一致） ======
    records = [
        {
            "timestamp": float(img_ts[i]),
            "clamp_left": (
                None
                if left_serial is None or aligned_clamp_by_serial.get(left_serial) is None
                else float(aligned_clamp_by_serial[left_serial][i])
            ),
            "clamp_right": (
                None
                if right_serial is None or aligned_clamp_by_serial.get(right_serial) is None
                else float(aligned_clamp_by_serial[right_serial][i])
            ),
            "pose_left": (
                None
                if left_serial is None or aligned_pose_by_serial.get(left_serial) is None
                else aligned_pose_by_serial[left_serial][i].astype(float).tolist()
            ),
            "pose_right": (
                None
                if right_serial is None or aligned_pose_by_serial.get(right_serial) is None
                else aligned_pose_by_serial[right_serial][i].astype(float).tolist()
            ),
            "imu": {
                topic: (
                    None
                    if imu_aligned.get(topic) is None
                    else {
                        **imu_aligned[topic][i],
                        "imu_err": float(imu_err[topic][i]),
                    }
                )
                for topic in imu_topics
            },
            "align": {
                "ref_camera": ref_topic,
                "pose_left_topic": f"/xv_sdk/{left_serial}/slam/pose" if left_serial else None,
                "pose_right_topic": f"/xv_sdk/{right_serial}/slam/pose" if right_serial else None,
                "clamp_left_topic": f"/xv_sdk/{left_serial}/clamp/Data" if left_serial else None,
                "clamp_right_topic": f"/xv_sdk/{right_serial}/clamp/Data" if right_serial else None,
                "pose_left_err": (
                    None
                    if left_serial is None or pose_err_by_serial.get(left_serial) is None
                    else float(pose_err_by_serial[left_serial][i])
                ),
                "pose_right_err": (
                    None
                    if right_serial is None or pose_err_by_serial.get(right_serial) is None
                    else float(pose_err_by_serial[right_serial][i])
                ),
                "clamp_left_err": (
                    None
                    if left_serial is None or clamp_err_by_serial.get(left_serial) is None
                    else float(clamp_err_by_serial[left_serial][i])
                ),
                "clamp_right_err": (
                    None
                    if right_serial is None or clamp_err_by_serial.get(right_serial) is None
                    else float(clamp_err_by_serial[right_serial][i])
                ),
            },
        }
        for i in range(len(img_ts))
    ]

    out_json = out_dir / f"{prefix}.json"
    out_json.write_text(
        json.dumps(
            {
                "fps": fps,
                "records": records,
                "videos": video_meta,
                "depth": depth_meta,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    plot_path = out_dir / f"{prefix}_states.png"
    plot_states(plot_pose, plot_clamp, plot_path)
    
    print(f"OK -> {out_dir}")
    print(f"OK -> {out_json}")
    print("Videos:")
    for k, v in video_meta.items():
        print(f"  {k}: {v['frame_count']} frames -> {v['file']}")


if __name__ == "__main__":
    main()
