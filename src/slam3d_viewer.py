"""
3D SLAM viewer - browser frontend.

Runs the same RPLIDAR + TFmini-S SLAM as slam3d_viewer.py, but instead of
matplotlib, serves a Three.js page over HTTP and streams point updates via
WebSocket. Way faster rendering and proper keyboard/mouse controls.

Usage:
    pip install -r requirements.txt  (now includes 'websockets')
    python slam3d_web.py
    # then open http://localhost:8000 in your browser

CLI flags same as slam3d_viewer.py: --lidar-port, --tfmini-port,
--tfmini-mode, --ceiling-height, --output.
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import webbrowser
from glob import glob
from queue import Queue, Empty

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d

import serial
import serial.tools.list_ports
from rplidar import RPLidar, RPLidarException

# Web stack
import http.server
import socketserver
import websockets


# ---------- RPLIDAR config ----------
RPLIDAR_BAUD = 115200
SCAN_TIMEOUT = 3
SCAN_QUEUE_MAX = 4

# ---------- TFmini-S config ----------
TFMINI_BAUD = 115200
TFMINI_HEADER = 0x59
TFMINI_FRAME_SIZE = 9
TFMINI_MIN_MM = 100
TFMINI_MAX_MM = 4000
HEIGHT_SMOOTHING = 0.3

DEFAULT_TFMINI_MODE = "floor"
DEFAULT_CEILING_MM = 2400

# ---------- SLAM config ----------
MIN_RANGE_MM = 150
MAX_RANGE_MM = 8000
DOWNSAMPLE_GRID_MM = 50

ICP_MAX_ITERATIONS = 25
ICP_TOLERANCE = 1e-4
ICP_MAX_PAIR_DIST_MM = 500

REF_CLOUD_MAX_POINTS = 8000
KEYFRAME_DIST_MM = 200
KEYFRAME_ROT_DEG = 10
KEYFRAME_Z_MM = 100

HIST_BINS = 360
HIST_BLUR_SIGMA = 2.0
HIST_MIN_OVERLAP_PTS = 50

# ---------- Server config ----------
HTTP_PORT = 8000
WS_PORT = 8001
POSE_BROADCAST_HZ = 20    # pose/stats updates per second


# ===================================================================
#  Serial helpers (same as slam3d_viewer.py)
# ===================================================================

def list_usbserial_ports():
    ports = []
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if any(kw in (desc + hwid) for kw in
               ["usb", "serial", "uart", "ftdi", "ch340", "cp210", "pl2303"]):
            ports.append(p.device)
    return sorted(set(ports))


def probe_tfmini(port, timeout=1.0):
    try:
        ser = serial.Serial(port, TFMINI_BAUD, timeout=0.3)
    except Exception:
        return False
    try:
        time.sleep(0.1)
        ser.reset_input_buffer()
        deadline = time.time() + timeout
        buf = b""
        while time.time() < deadline:
            chunk = ser.read(64)
            if chunk:
                buf += chunk
                if b"\x59\x59" in buf:
                    return True
                if len(buf) > 512:
                    buf = buf[-64:]
        return False
    finally:
        ser.close()


def probe_rplidar(port, timeout=1.5):
    try:
        lidar = RPLidar(port, baudrate=RPLIDAR_BAUD, timeout=timeout)
        info = lidar.get_info()
        lidar.disconnect()
        return "model" in info
    except Exception:
        return False


def auto_detect_ports(explicit_lidar=None, explicit_tfmini=None):
    lidar_port = explicit_lidar
    tfmini_port = explicit_tfmini
    if lidar_port and tfmini_port:
        return lidar_port, tfmini_port

    candidates = list_usbserial_ports()
    if lidar_port and lidar_port in candidates:
        candidates.remove(lidar_port)
    if tfmini_port and tfmini_port in candidates:
        candidates.remove(tfmini_port)

    if not candidates:
        sys.exit("ERROR: no USB-serial ports found.")
    if len(candidates) < 2 and not (lidar_port or tfmini_port):
        sys.exit(f"ERROR: need 2 USB-serial adapters, only found: {candidates}")

    if not tfmini_port:
        print("Probing for TFmini-S...")
        for c in list(candidates):
            if probe_tfmini(c):
                tfmini_port = c
                candidates.remove(c)
                print(f"  TFmini-S found on {c}")
                break
        if not tfmini_port:
            sys.exit("ERROR: could not find TFmini-S. Pass --tfmini-port explicitly.")

    if not lidar_port:
        print("Probing for RPLIDAR...")
        for c in candidates:
            if probe_rplidar(c):
                lidar_port = c
                print(f"  RPLIDAR found on {c}")
                break
        if not lidar_port:
            if candidates:
                lidar_port = candidates[0]
                print(f"  Assuming RPLIDAR on {lidar_port} (probe failed)")
            else:
                sys.exit("ERROR: could not find RPLIDAR. Pass --lidar-port explicitly.")

    return lidar_port, tfmini_port


# ===================================================================
#  Scan / cloud / ICP math (same as before)
# ===================================================================

def scan_to_points(scan):
    pts = []
    for (_q, ang_deg, dist) in scan:
        if MIN_RANGE_MM <= dist <= MAX_RANGE_MM:
            a = np.deg2rad(ang_deg)
            pts.append((dist * np.sin(a), dist * np.cos(a)))
    return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 2))


def voxel_downsample_2d(points, grid_mm):
    if len(points) == 0:
        return points
    keys = np.floor(points / grid_mm).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


def voxel_downsample_3d(points, grid_mm):
    if len(points) == 0:
        return points
    keys = np.floor(points / grid_mm).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


def apply_transform_2d(points, dx, dy, dtheta):
    if len(points) == 0:
        return points
    c, s = np.cos(dtheta), np.sin(dtheta)
    R = np.array([[c, -s], [s, c]])
    return points @ R.T + np.array([dx, dy])


def angle_histogram(points, bins=HIST_BINS):
    if len(points) == 0:
        return np.zeros(bins)
    ang = (np.arctan2(points[:, 1], points[:, 0]) + 2 * np.pi) % (2 * np.pi)
    dist = np.linalg.norm(points, axis=1)
    bin_idx = (ang / (2 * np.pi) * bins).astype(int) % bins
    hist = np.zeros(bins)
    counts = np.zeros(bins)
    np.add.at(hist, bin_idx, dist)
    np.add.at(counts, bin_idx, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hist = np.where(counts > 0, hist / np.maximum(counts, 1), 0)
    if HIST_BLUR_SIGMA > 0:
        hist = gaussian_filter1d(hist, sigma=HIST_BLUR_SIGMA, mode="wrap")
    return hist


def estimate_rotation(source_pts, target_pts):
    if len(source_pts) < HIST_MIN_OVERLAP_PTS or len(target_pts) < HIST_MIN_OVERLAP_PTS:
        return 0.0
    h_src = angle_histogram(source_pts)
    h_tgt = angle_histogram(target_pts)
    h_src -= h_src.mean()
    h_tgt -= h_tgt.mean()
    corr = np.fft.irfft(np.fft.rfft(h_tgt) * np.conj(np.fft.rfft(h_src)),
                        n=HIST_BINS)
    best_shift = int(np.argmax(corr))
    dtheta = best_shift * (2 * np.pi / HIST_BINS)
    if dtheta > np.pi:
        dtheta -= 2 * np.pi
    return dtheta


def icp(source, target, init_dx=0.0, init_dy=0.0, init_dtheta=0.0,
        max_iter=ICP_MAX_ITERATIONS, tol=ICP_TOLERANCE,
        max_pair_dist=ICP_MAX_PAIR_DIST_MM):
    if len(source) < 5 or len(target) < 5:
        return init_dx, init_dy, init_dtheta, float("inf")
    tree = cKDTree(target)
    dx, dy, dtheta = init_dx, init_dy, init_dtheta
    prev_error = float("inf")
    for _ in range(max_iter):
        src_t = apply_transform_2d(source, dx, dy, dtheta)
        dists, idxs = tree.query(src_t, k=1, workers=-1)
        mask = dists < max_pair_dist
        if mask.sum() < 5:
            break
        src_pairs = src_t[mask]
        tgt_pairs = target[idxs[mask]]
        sc = src_pairs.mean(axis=0)
        tc = tgt_pairs.mean(axis=0)
        src_c = src_pairs - sc
        tgt_c = tgt_pairs - tc
        H = src_c.T @ tgt_c
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = tc - R @ sc
        ddtheta = np.arctan2(R[1, 0], R[0, 0])
        dx, dy = R @ np.array([dx, dy]) + t
        dtheta += ddtheta
        error = dists[mask].mean()
        if abs(prev_error - error) < tol * max_pair_dist:
            break
        prev_error = error
    return dx, dy, dtheta, prev_error


# ===================================================================
#  Sensor threads (same as before)
# ===================================================================

class LidarThread(threading.Thread):
    def __init__(self, lidar, scan_queue):
        super().__init__(daemon=True)
        self.lidar = lidar
        self.queue = scan_queue
        self.running = True

    def run(self):
        try:
            for scan in self.lidar.iter_scans(max_buf_meas=2000, min_len=5):
                if not self.running:
                    break
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except Empty:
                        pass
                self.queue.put(scan)
        except RPLidarException as e:
            print(f"LiDAR thread error: {e}")
        except Exception as e:
            print(f"LiDAR thread crashed: {e}")

    def stop(self):
        self.running = False


class TFminiThread(threading.Thread):
    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port
        self.ser = None
        self.running = True
        self._lock = threading.Lock()
        self._distance_mm = 1000.0
        self._last_good_time = 0.0
        self.frames_total = 0
        self.frames_good = 0

    @property
    def distance_mm(self):
        with self._lock:
            return self._distance_mm

    @property
    def is_stale(self):
        return (time.time() - self._last_good_time) > 0.5

    def _read_frame(self):
        while self.running:
            b = self.ser.read(1)
            if not b:
                return None
            if b[0] == TFMINI_HEADER:
                b2 = self.ser.read(1)
                if not b2:
                    return None
                if b2[0] == TFMINI_HEADER:
                    break
        rest = self.ser.read(7)
        if len(rest) < 7:
            return None
        frame = bytes([TFMINI_HEADER, TFMINI_HEADER]) + rest
        if (sum(frame[:8]) & 0xFF) != frame[8]:
            return None
        dist_cm = frame[2] | (frame[3] << 8)
        return dist_cm * 10

    def run(self):
        try:
            self.ser = serial.Serial(self.port, TFMINI_BAUD, timeout=1)
            time.sleep(0.1)
            self.ser.reset_input_buffer()
        except Exception as e:
            print(f"TFmini open failed: {e}")
            return
        while self.running:
            dist_mm = self._read_frame()
            if dist_mm is None:
                continue
            self.frames_total += 1
            if not (TFMINI_MIN_MM <= dist_mm <= TFMINI_MAX_MM):
                continue
            self.frames_good += 1
            with self._lock:
                self._distance_mm += HEIGHT_SMOOTHING * (dist_mm - self._distance_mm)
                self._last_good_time = time.time()
        try:
            self.ser.close()
        except Exception:
            pass

    def stop(self):
        self.running = False


# ===================================================================
#  Shared state - SLAM and frontend talk through this
# ===================================================================

class SharedState:
    """All cross-thread mutable state. Lock when touching collections."""
    def __init__(self, mode="floor", ceiling_mm=2400):
        self.lock = threading.Lock()
        self.pose = np.array([0.0, 0.0, 0.0])   # x_mm, y_mm, theta_rad
        self.z_mm = 1000.0                       # sensor height above floor
        self.trajectory = []                     # list of (x_m, y_m, z_m)
        self.map_points_3d = []                  # list of (N, 3) arrays in world mm
        self.ref_cloud_2d = np.zeros((0, 2))
        self.last_kf_pose = self.pose.copy()
        self.last_kf_z_mm = None
        # Outgoing point batches that haven't been broadcast yet
        # Each entry: dict with keys 'points' (Nx3 mm) and 'keyframe_idx'
        self.pending_broadcasts = []
        # Counters
        self.scan_count = 0
        self.keyframe_count = 0
        # Settings (mutable at runtime)
        self.tfmini_mode = mode            # "floor" or "ceiling"
        self.ceiling_mm = ceiling_mm
        self.recording = True              # spacebar / record button toggles this
        self.tfmini_distance_mm = 0.0      # raw TFmini reading (for display)
        self.tfmini_stale = False


# ===================================================================
#  SLAM worker thread - same logic as slam3d_viewer.py
# ===================================================================

class SlamWorker(threading.Thread):
    def __init__(self, scan_queue, tfmini_thread, state, output_path):
        super().__init__(daemon=True)
        self.scan_queue = scan_queue
        self.tfmini = tfmini_thread
        self.state = state
        self.output_path = output_path
        self.running = True

    def stop(self):
        self.running = False

    def _sensor_height_mm(self):
        d = self.tfmini.distance_mm
        if self.state.tfmini_mode == "floor":
            return d
        return max(0.0, self.state.ceiling_mm - d)

    def run(self):
        while self.running:
            try:
                scan = self.scan_queue.get(timeout=0.1)
            except Empty:
                continue

            s = self.state
            s.scan_count += 1
            s.tfmini_distance_mm = self.tfmini.distance_mm
            s.tfmini_stale = self.tfmini.is_stale
            z_mm = self._sensor_height_mm()
            s.z_mm = z_mm

            pts_sensor = scan_to_points(scan)
            pts_sensor = voxel_downsample_2d(pts_sensor, DOWNSAMPLE_GRID_MM)
            if len(pts_sensor) < 10:
                continue

            # 2D ICP
            if len(s.ref_cloud_2d) > 50:
                ref_in_prev = apply_transform_2d(s.ref_cloud_2d, -s.pose[0], -s.pose[1], 0.0)
                ref_in_prev = apply_transform_2d(ref_in_prev, 0.0, 0.0, -s.pose[2])
                ddtheta = estimate_rotation(pts_sensor, ref_in_prev)
                init_theta = s.pose[2] + ddtheta
                dx, dy, dtheta, _err = icp(
                    pts_sensor, s.ref_cloud_2d,
                    init_dx=s.pose[0], init_dy=s.pose[1], init_dtheta=init_theta)
                s.pose = np.array([dx, dy, dtheta])

            # World-frame 3D points
            pts_world_2d = apply_transform_2d(pts_sensor, s.pose[0], s.pose[1], s.pose[2])
            z_col = np.full((len(pts_world_2d), 1), z_mm)
            pts_world_3d = np.hstack([pts_world_2d, z_col])

            # Trajectory (always updated, regardless of recording state)
            s.trajectory.append((s.pose[0]/1000.0, s.pose[1]/1000.0, z_mm/1000.0))

            # Keyframe decision - ONLY commits points if recording is on
            if not s.recording:
                continue

            d_trans = np.linalg.norm(s.pose[:2] - s.last_kf_pose[:2])
            d_rot = abs(np.rad2deg((s.pose[2] - s.last_kf_pose[2] + np.pi)
                                   % (2*np.pi) - np.pi))
            d_z = abs(z_mm - s.last_kf_z_mm) if s.last_kf_z_mm is not None else float("inf")
            is_first = len(s.map_points_3d) == 0
            if (is_first or d_trans > KEYFRAME_DIST_MM or d_rot > KEYFRAME_ROT_DEG
                    or d_z > KEYFRAME_Z_MM):
                with s.lock:
                    s.map_points_3d.append(pts_world_3d)
                    s.last_kf_pose = s.pose.copy()
                    s.last_kf_z_mm = z_mm
                    s.keyframe_count += 1

                    # Update ICP reference cloud (whole-map, downsampled)
                    all_pts_2d = np.vstack([p[:, :2] for p in s.map_points_3d])
                    if len(all_pts_2d) > REF_CLOUD_MAX_POINTS:
                        idx = np.random.choice(len(all_pts_2d), REF_CLOUD_MAX_POINTS,
                                               replace=False)
                        s.ref_cloud_2d = all_pts_2d[idx]
                    else:
                        s.ref_cloud_2d = all_pts_2d

                    # Queue this keyframe for broadcast to browser
                    # Downsample to ~5cm for transport (full res stays in map_points_3d)
                    transport_pts = voxel_downsample_3d(pts_world_3d, 50)
                    s.pending_broadcasts.append({
                        "points_m": (transport_pts / 1000.0).astype(np.float32),
                        "keyframe_idx": s.keyframe_count - 1,
                    })

    def reset_map(self):
        s = self.state
        with s.lock:
            s.pose = np.array([0.0, 0.0, 0.0])
            s.trajectory = []
            s.map_points_3d = []
            s.ref_cloud_2d = np.zeros((0, 2))
            s.last_kf_pose = s.pose.copy()
            s.last_kf_z_mm = None
            s.pending_broadcasts = []
            s.scan_count = 0
            s.keyframe_count = 0

    def save_now(self, custom_path=None):
        s = self.state
        path = custom_path or self.output_path
        with s.lock:
            if not s.map_points_3d:
                return None, "no points yet"
            all_pts = np.vstack(s.map_points_3d) / 1000.0
            traj = np.array(s.trajectory) if s.trajectory else np.zeros((0, 3))
            np.savez(path, points=all_pts, trajectory=traj,
                     scan_count=s.scan_count, keyframe_count=s.keyframe_count)
            return path, f"{len(all_pts)} points, {len(traj)} trajectory samples"


# ===================================================================
#  HTML page (embedded so this is one file)
# ===================================================================

INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>3D SLAM Viewer</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; background: #0d1117;
               color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont,
                                            "Segoe UI", sans-serif; }
  #viewer { position: fixed; top: 0; left: 0; right: 320px; bottom: 0; }
  #panel { position: fixed; top: 0; right: 0; bottom: 0; width: 320px;
           background: #161b22; border-left: 1px solid #30363d;
           padding: 20px; overflow-y: auto; }
  h2 { margin: 0 0 12px; font-size: 16px; font-weight: 600; color: #e6edf3; }
  h3 { margin: 24px 0 8px; font-size: 12px; font-weight: 600;
       color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
  .stat { display: flex; justify-content: space-between; padding: 6px 0;
          font-size: 13px; border-bottom: 1px solid #21262d; }
  .stat .k { color: #8b949e; }
  .stat .v { color: #e6edf3; font-family: ui-monospace, "SF Mono", monospace; }
  .v.warn { color: #f0883e; }
  button { display: block; width: 100%; padding: 10px 12px; margin: 6px 0;
           background: #21262d; color: #e6edf3; border: 1px solid #30363d;
           border-radius: 6px; cursor: pointer; font-size: 13px;
           text-align: left; }
  button:hover { background: #2d333b; border-color: #444c56; }
  button .key { float: right; color: #8b949e; font-family: monospace;
                font-size: 11px; }
  button.record { background: #b62324; border-color: #b62324; color: #fff; }
  button.record:hover { background: #cf2729; }
  button.record.paused { background: #21262d; border-color: #30363d;
                         color: #e6edf3; }
  .rec-indicator { display: inline-block; width: 10px; height: 10px;
                   border-radius: 50%; margin-right: 8px; vertical-align: middle;
                   background: #b62324; }
  .rec-indicator.paused { background: #6e7681; }
  .rec-indicator.recording { animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  #status { margin-bottom: 12px; padding: 10px 12px;
            background: #21262d; border-radius: 6px; font-size: 13px; }
  .help { margin-top: 24px; font-size: 11px; color: #6e7681; line-height: 1.6; }
  .help b { color: #8b949e; }
</style>
</head>
<body>

<div id="viewer"></div>

<div id="panel">
  <h2>3D SLAM Viewer</h2>

  <div id="status">
    <span id="recIndicator" class="rec-indicator recording"></span>
    <span id="recText">Recording</span>
  </div>

  <button id="btnRecord" class="record">
    Pause recording <span class="key">SPACE</span>
  </button>

  <h3>Stats</h3>
  <div class="stat"><span class="k">Scans</span><span class="v" id="vScans">0</span></div>
  <div class="stat"><span class="k">Keyframes</span><span class="v" id="vKfs">0</span></div>
  <div class="stat"><span class="k">Points</span><span class="v" id="vPts">0</span></div>
  <div class="stat"><span class="k">Pose X</span><span class="v" id="vX">0.00 m</span></div>
  <div class="stat"><span class="k">Pose Y</span><span class="v" id="vY">0.00 m</span></div>
  <div class="stat"><span class="k">Pose Z</span><span class="v" id="vZ">0.00 m</span></div>
  <div class="stat"><span class="k">TFmini raw</span><span class="v" id="vRaw">0.00 m</span></div>
  <div class="stat"><span class="k">Mode</span><span class="v" id="vMode">floor</span></div>

  <h3>Actions</h3>
  <button id="btnSave">Save now <span class="key">S</span></button>
  <button id="btnReset">Reset map <span class="key">R</span></button>
  <button id="btnFit">Fit view <span class="key">F</span></button>
  <button id="btnMode">Toggle floor / ceiling <span class="key">C</span></button>
  <button id="btnCeilDown">Lower ceiling 10cm <span class="key">[</span></button>
  <button id="btnCeilUp">Raise ceiling 10cm <span class="key">]</span></button>

  <div class="help">
    <b>Mouse:</b> drag to orbit, scroll to zoom, right-drag to pan.<br>
    <b>Recording paused:</b> sensors stay live and the red pose dot still
    moves, but no new points get added to the map.
  </div>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ---------- Three.js setup ----------
const viewer = document.getElementById('viewer');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.PerspectiveCamera(
  60, viewer.clientWidth / viewer.clientHeight, 0.05, 200);
camera.up.set(0, 0, 1);   // Z is up
camera.position.set(5, -5, 4);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(viewer.clientWidth, viewer.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);
viewer.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 1);
controls.update();

// Grid on the floor (XY plane at z=0)
const grid = new THREE.GridHelper(20, 20, 0x2a2f3a, 0x21262d);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

// Axes for reference (small)
const axes = new THREE.AxesHelper(0.5);
scene.add(axes);

// Points object - we grow this as new keyframes arrive
const MAX_POINTS = 500000;
const pointsGeometry = new THREE.BufferGeometry();
const positions = new Float32Array(MAX_POINTS * 3);
const colors = new Float32Array(MAX_POINTS * 3);
pointsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
pointsGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
pointsGeometry.setDrawRange(0, 0);
const pointsMaterial = new THREE.PointsMaterial({
  size: 0.03,
  vertexColors: true,
  sizeAttenuation: true,
});
const pointsObject = new THREE.Points(pointsGeometry, pointsMaterial);
scene.add(pointsObject);
let nextPointIdx = 0;

// Trajectory line
const trajGeometry = new THREE.BufferGeometry();
const MAX_TRAJ = 100000;
const trajPositions = new Float32Array(MAX_TRAJ * 3);
trajGeometry.setAttribute('position', new THREE.BufferAttribute(trajPositions, 3));
trajGeometry.setDrawRange(0, 0);
const trajLine = new THREE.Line(trajGeometry,
  new THREE.LineBasicMaterial({ color: 0x39d0d8 }));
scene.add(trajLine);
let nextTrajIdx = 0;

// Pose marker (red sphere)
const poseMarker = new THREE.Mesh(
  new THREE.SphereGeometry(0.08, 16, 16),
  new THREE.MeshBasicMaterial({ color: 0xff4444 })
);
scene.add(poseMarker);

// Viridis-ish color ramp for Z (0m = dark purple, 3m = yellow)
function zToColor(z) {
  // Approximate viridis: 5 control points
  const stops = [
    [0.000, 0.267, 0.005, 0.329],  // dark purple
    [0.250, 0.275, 0.190, 0.490],
    [0.500, 0.127, 0.567, 0.550],  // teal
    [0.750, 0.369, 0.788, 0.382],
    [1.000, 0.992, 0.906, 0.144],  // yellow
  ];
  const t = Math.max(0, Math.min(1, z / 3.0));
  for (let i = 0; i < stops.length - 1; i++) {
    if (t <= stops[i+1][0]) {
      const f = (t - stops[i][0]) / (stops[i+1][0] - stops[i][0]);
      return [
        stops[i][1] + f * (stops[i+1][1] - stops[i][1]),
        stops[i][2] + f * (stops[i+1][2] - stops[i][2]),
        stops[i][3] + f * (stops[i+1][3] - stops[i][3]),
      ];
    }
  }
  return [stops[stops.length-1][1], stops[stops.length-1][2], stops[stops.length-1][3]];
}

function addPoints(pts) {
  // pts is array of [x, y, z] in meters
  const posAttr = pointsGeometry.attributes.position;
  const colAttr = pointsGeometry.attributes.color;
  for (let i = 0; i < pts.length; i++) {
    if (nextPointIdx >= MAX_POINTS) break;
    const [x, y, z] = pts[i];
    posAttr.array[nextPointIdx * 3]     = x;
    posAttr.array[nextPointIdx * 3 + 1] = y;
    posAttr.array[nextPointIdx * 3 + 2] = z;
    const [r, g, b] = zToColor(z);
    colAttr.array[nextPointIdx * 3]     = r;
    colAttr.array[nextPointIdx * 3 + 1] = g;
    colAttr.array[nextPointIdx * 3 + 2] = b;
    nextPointIdx++;
  }
  posAttr.needsUpdate = true;
  colAttr.needsUpdate = true;
  pointsGeometry.setDrawRange(0, nextPointIdx);
}

function setTrajectory(traj) {
  // traj: array of [x, y, z]. We replace whole line each update for simplicity.
  const arr = trajGeometry.attributes.position.array;
  const n = Math.min(traj.length, MAX_TRAJ);
  for (let i = 0; i < n; i++) {
    arr[i*3]   = traj[i][0];
    arr[i*3+1] = traj[i][1];
    arr[i*3+2] = traj[i][2];
  }
  trajGeometry.attributes.position.needsUpdate = true;
  trajGeometry.setDrawRange(0, n);
}

function setPose(x, y, z) {
  poseMarker.position.set(x, y, z);
}

function clearAll() {
  nextPointIdx = 0;
  pointsGeometry.setDrawRange(0, 0);
  trajGeometry.setDrawRange(0, 0);
  poseMarker.position.set(0, 0, 0);
}

function fitView() {
  // Simple reset
  controls.target.set(0, 0, 1);
  camera.position.set(5, -5, 4);
  controls.update();
}

window.addEventListener('resize', () => {
  camera.aspect = viewer.clientWidth / viewer.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(viewer.clientWidth, viewer.clientHeight);
});

function render() {
  requestAnimationFrame(render);
  controls.update();
  renderer.render(scene, camera);
}
render();

// ---------- WebSocket ----------
const wsUrl = `ws://${location.hostname}:%%WS_PORT%%`;
let ws;
function connect() {
  ws = new WebSocket(wsUrl);
  ws.onopen = () => console.log('WS connected');
  ws.onclose = () => {
    console.log('WS disconnected, retrying in 1s');
    setTimeout(connect, 1000);
  };
  ws.onerror = (e) => console.error('WS error', e);
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'pose') {
      document.getElementById('vScans').textContent = msg.scans;
      document.getElementById('vKfs').textContent   = msg.keyframes;
      document.getElementById('vPts').textContent   = nextPointIdx;
      document.getElementById('vX').textContent = msg.x.toFixed(2) + ' m';
      document.getElementById('vY').textContent = msg.y.toFixed(2) + ' m';
      document.getElementById('vZ').textContent = msg.z.toFixed(2) + ' m';
      const raw = document.getElementById('vRaw');
      raw.textContent = msg.raw.toFixed(2) + ' m';
      raw.className = 'v' + (msg.stale ? ' warn' : '');
      const modeEl = document.getElementById('vMode');
      modeEl.textContent = msg.mode + (msg.mode === 'ceiling'
        ? ` (${msg.ceiling.toFixed(2)}m)` : '');
      setPose(msg.x, msg.y, msg.z);
      if (msg.trajectory) setTrajectory(msg.trajectory);
      setRecording(msg.recording);
    } else if (msg.type === 'points') {
      addPoints(msg.points);
    } else if (msg.type === 'reset') {
      clearAll();
    } else if (msg.type === 'saved') {
      flash(`Saved: ${msg.path} (${msg.info})`);
    } else if (msg.type === 'error') {
      flash(`Error: ${msg.msg}`, true);
    }
  };
}
connect();

function send(cmd, extra) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(Object.assign({ cmd }, extra || {})));
  }
}

function setRecording(recording) {
  const ind = document.getElementById('recIndicator');
  const txt = document.getElementById('recText');
  const btn = document.getElementById('btnRecord');
  if (recording) {
    ind.className = 'rec-indicator recording';
    txt.textContent = 'Recording';
    btn.classList.add('record');
    btn.classList.remove('paused');
    btn.innerHTML = 'Pause recording <span class="key">SPACE</span>';
  } else {
    ind.className = 'rec-indicator paused';
    txt.textContent = 'Paused';
    btn.classList.add('paused');
    btn.classList.remove('record');
    btn.innerHTML = 'Resume recording <span class="key">SPACE</span>';
  }
}

function flash(text, isError) {
  const s = document.getElementById('status');
  const orig = s.innerHTML;
  s.innerHTML = `<span style="color:${isError ? '#f85149' : '#3fb950'}">${text}</span>`;
  setTimeout(() => { s.innerHTML = orig; }, 2500);
}

// ---------- Buttons ----------
document.getElementById('btnRecord').onclick   = () => send('toggle_recording');
document.getElementById('btnSave').onclick     = () => send('save');
document.getElementById('btnReset').onclick    = () => {
  if (confirm('Clear all points and trajectory?')) send('reset');
};
document.getElementById('btnFit').onclick      = () => fitView();
document.getElementById('btnMode').onclick     = () => send('toggle_mode');
document.getElementById('btnCeilDown').onclick = () => send('ceiling_delta', { delta_mm: -100 });
document.getElementById('btnCeilUp').onclick   = () => send('ceiling_delta', { delta_mm:  100 });

// ---------- Keyboard ----------
window.addEventListener('keydown', (e) => {
  // Skip if user is typing in an input (none right now, but future-proof)
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  switch (e.key) {
    case ' ':  e.preventDefault(); send('toggle_recording'); break;
    case 's':  send('save'); break;
    case 'r':  if (confirm('Clear all points and trajectory?')) send('reset'); break;
    case 'f':  fitView(); break;
    case 'c':  send('toggle_mode'); break;
    case '[':  send('ceiling_delta', { delta_mm: -100 }); break;
    case ']':  send('ceiling_delta', { delta_mm:  100 }); break;
  }
});

</script>
</body>
</html>
"""


# ===================================================================
#  HTTP server (serves the index.html)
# ===================================================================

class IndexHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = INDEX_HTML.replace("%%WS_PORT%%", str(WS_PORT))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        return  # silence access logs


def serve_http():
    with socketserver.ThreadingTCPServer(("", HTTP_PORT), IndexHandler) as httpd:
        httpd.allow_reuse_address = True
        httpd.serve_forever()


# ===================================================================
#  WebSocket server (broadcast loop + command handler)
# ===================================================================

clients = set()

async def ws_handler(websocket, slam_worker, state):
    clients.add(websocket)
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            await handle_command(msg, websocket, slam_worker, state)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        clients.discard(websocket)


async def handle_command(msg, websocket, slam_worker, state):
    cmd = msg.get("cmd")
    if cmd == "toggle_recording":
        state.recording = not state.recording
        print(f"  -> recording: {state.recording}")
    elif cmd == "save":
        path, info = slam_worker.save_now()
        if path is None:
            await websocket.send(json.dumps({"type": "error", "msg": info}))
        else:
            print(f"  -> saved to {path}")
            await broadcast({"type": "saved", "path": path, "info": info})
    elif cmd == "reset":
        slam_worker.reset_map()
        print("  -> map reset")
        await broadcast({"type": "reset"})
    elif cmd == "toggle_mode":
        state.tfmini_mode = "ceiling" if state.tfmini_mode == "floor" else "floor"
        print(f"  -> tfmini mode: {state.tfmini_mode}")
    elif cmd == "ceiling_delta":
        delta = msg.get("delta_mm", 0)
        state.ceiling_mm = max(500, state.ceiling_mm + delta)
        print(f"  -> ceiling: {state.ceiling_mm/1000:.2f}m")


async def broadcast(msg):
    if not clients:
        return
    payload = json.dumps(msg)
    dead = []
    for ws in clients:
        try:
            await ws.send(payload)
        except Exception:
            dead.append(ws)
    for d in dead:
        clients.discard(d)


async def broadcast_loop(state):
    """Send pose+stats at fixed rate, and drain pending point batches."""
    interval = 1.0 / POSE_BROADCAST_HZ
    last_traj_len = 0
    while True:
        await asyncio.sleep(interval)
        if not clients:
            continue

        # Drain pending point batches (built by SLAM worker)
        with state.lock:
            pending = state.pending_broadcasts
            state.pending_broadcasts = []

        for batch in pending:
            pts = batch["points_m"].tolist()
            await broadcast({"type": "points", "points": pts})

        # Trajectory delta: send full traj every few seconds, or whenever
        # it grew a lot. Simpler: just send the whole thing - it's small.
        with state.lock:
            traj = list(state.trajectory)
            scans = state.scan_count
            kfs = state.keyframe_count
            pose = state.pose.copy()
            z_mm = state.z_mm
            mode = state.tfmini_mode
            ceil = state.ceiling_mm
            raw_mm = state.tfmini_distance_mm
            stale = state.tfmini_stale
            recording = state.recording

        await broadcast({
            "type": "pose",
            "scans": scans,
            "keyframes": kfs,
            "x": pose[0]/1000.0,
            "y": pose[1]/1000.0,
            "z": z_mm/1000.0,
            "mode": mode,
            "ceiling": ceil/1000.0,
            "raw": raw_mm/1000.0,
            "stale": stale,
            "recording": recording,
            "trajectory": traj if len(traj) != last_traj_len else None,
        })
        last_traj_len = len(traj)


async def run_websocket_server(slam_worker, state):
    async def handler(ws):
        await ws_handler(ws, slam_worker, state)
    async with websockets.serve(handler, "", WS_PORT):
        await broadcast_loop(state)


# ===================================================================
#  Main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(description="3D SLAM viewer (browser frontend)")
    p.add_argument("--lidar-port", default=None)
    p.add_argument("--tfmini-port", default=None)
    p.add_argument("--tfmini-mode", choices=["floor", "ceiling"],
                   default=DEFAULT_TFMINI_MODE)
    p.add_argument("--ceiling-height", type=float,
                   default=DEFAULT_CEILING_MM/1000.0,
                   help="Ceiling height in meters (default 2.4)")
    p.add_argument("--output", default="apartment_3d.npz")
    p.add_argument("--no-browser", action="store_true",
                   help="Don't auto-open the browser")
    return p.parse_args()


def main():
    args = parse_args()
    lidar_port, tfmini_port = auto_detect_ports(args.lidar_port, args.tfmini_port)
    print(f"RPLIDAR: {lidar_port}")
    print(f"TFmini-S: {tfmini_port}")

    tfmini_thread = TFminiThread(tfmini_port)
    tfmini_thread.start()
    print("Waiting for TFmini...")
    t0 = time.time()
    while tfmini_thread.frames_good < 5 and time.time() - t0 < 3.0:
        time.sleep(0.1)
    if tfmini_thread.frames_good < 5:
        print("WARNING: TFmini not producing good data yet.")
    else:
        print(f"  Initial TFmini distance: {tfmini_thread.distance_mm/1000:.2f} m")

    print("Connecting to RPLIDAR...")
    lidar = RPLidar(lidar_port, baudrate=RPLIDAR_BAUD, timeout=SCAN_TIMEOUT)
    try:
        print(f"  Info: {lidar.get_info()}")
        print(f"  Health: {lidar.get_health()}")
    except RPLidarException as e:
        lidar.stop(); lidar.disconnect(); tfmini_thread.stop()
        sys.exit(f"LiDAR error: {e}")

    scan_queue = Queue(maxsize=SCAN_QUEUE_MAX)
    lidar_thread = LidarThread(lidar, scan_queue)
    lidar_thread.start()

    state = SharedState(mode=args.tfmini_mode,
                        ceiling_mm=int(args.ceiling_height * 1000))
    slam_worker = SlamWorker(scan_queue, tfmini_thread, state, args.output)
    slam_worker.start()

    # HTTP server in a daemon thread
    http_thread = threading.Thread(target=serve_http, daemon=True)
    http_thread.start()

    url = f"http://localhost:{HTTP_PORT}"
    print(f"\n  >> Open {url} in your browser  <<")
    print("  WebSocket on ws://localhost:%d\n" % WS_PORT)
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    print("Press Ctrl+C in this terminal to stop.\n")

    try:
        asyncio.run(run_websocket_server(slam_worker, state))
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        slam_worker.stop()
        lidar_thread.stop()
        tfmini_thread.stop()
        time.sleep(0.5)
        try:
            lidar.stop(); lidar.stop_motor(); lidar.disconnect()
        except Exception:
            pass

        # Final save on exit
        path, info = slam_worker.save_now()
        if path:
            print(f"Final save -> {path}  ({info})")
        print(f"Total scans: {state.scan_count}, keyframes: {state.keyframe_count}")


if __name__ == "__main__":
    main()