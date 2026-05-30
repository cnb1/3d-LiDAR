"""
BNO055 IMU test viewer - via Pololu USB-to-I2C adapter.

Reads orientation from an Adafruit BNO055 connected via Pololu's
USB-to-I2C Adapter (with Isolated Power), and shows a live 3D rendering
in a browser. Use this to verify the sensor works and is calibrated
before integrating with the SLAM viewer.

Hardware:
    Mac USB-C <-> Pololu USB-to-I2C Adapter <-> STEMMA QT cable <-> BNO055

Why Pololu instead of FT232H:
    The BNO055 uses I2C clock stretching, which the FT232H does not
    support reliably. The Pololu adapter explicitly supports clock
    stretching at speeds in excess of 1 MHz.

Architecture:
    This script is structured so the Pololu adapter is treated as
    swappable scaffolding. The BNO055 driver above it sees a generic
    CircuitPython-style I2C bus, so when you eventually move to a
    Raspberry Pi or microcontroller for production, only the bus
    creation line changes.

Usage:
    python imu_test.py
    # Open http://localhost:8000 in your browser

    If the adapter shows up at a different port:
    python imu_test.py --port /dev/cu.usbmodem1401

Calibration (on every power-on):
    - GYRO: hold sensor still for a few seconds
    - ACCEL: tilt slowly through 6 orientations
    - MAG: rotate in a figure-8 in the air, away from metal
    - SYS: reaches 3 when the others all do
"""

import argparse
import asyncio
import http.server
import json
import socketserver
import sys
import threading
import time
import webbrowser

import websockets

import pololu_usb_i2c_adapter
import adafruit_bno055


# ===================================================================
#  Config
# ===================================================================

DEFAULT_PORT = "/dev/cu.usbmodem1401"
HTTP_PORT = 8000
WS_PORT = 8001
SAMPLE_HZ = 50           # how often to poll the BNO055
BROADCAST_HZ = 30        # how often to push to the browser

# BNO055 I2C address (0x29 if the ADR pad is bridged)
BNO055_ADDR = 0x28


# ===================================================================
#  Pololu -> CircuitPython I2C shim
# ===================================================================
#
# adafruit_bno055.BNO055_I2C expects a CircuitPython-style I2C bus
# object (busio.I2C-like). It calls these methods:
#   - try_lock()         -> bool
#   - unlock()           -> None
#   - writeto(addr, buf, *, start=0, end=None, stop=True)
#   - readfrom_into(addr, buf, *, start=0, end=None)
#   - writeto_then_readfrom(addr, out_buf, in_buf,
#                            out_start=0, out_end=None,
#                            in_start=0, in_end=None)
#
# pololu_usb_i2c_adapter.Adapter exposes:
#   - write_to(addr, data: bytes)
#   - read_from(addr, count: int) -> bytes
#
# This shim translates one to the other. In production (moving to a
# Raspberry Pi), this class disappears and you use busio.I2C() directly.

class PololuI2CShim:
    """Adapts a Pololu USB-to-I2C Adapter to the CircuitPython I2C API."""

    def __init__(self, port):
        self._adapter = pololu_usb_i2c_adapter.Adapter(port)
        # CircuitPython locks are reentrant within the same thread but
        # we keep it simple here - one user at a time.
        self._lock = threading.Lock()
        self._locked = False

    def try_lock(self):
        got = self._lock.acquire(blocking=False)
        if got:
            self._locked = True
        return got

    def unlock(self):
        if self._locked:
            self._locked = False
            self._lock.release()

    def writeto(self, addr, buf, *, start=0, end=None, stop=True):
        data = bytes(buf[start:end])
        self._adapter.write_to(addr, data)

    def readfrom_into(self, addr, buf, *, start=0, end=None):
        if end is None:
            end = len(buf)
        n = end - start
        data = self._adapter.read_from(addr, n)
        buf[start:end] = data

    def writeto_then_readfrom(self, addr, out_buf, in_buf,
                              out_start=0, out_end=None,
                              in_start=0, in_end=None):
        out_data = bytes(out_buf[out_start:out_end])
        if in_end is None:
            in_end = len(in_buf)
        n_in = in_end - in_start
        self._adapter.write_to(addr, out_data)
        in_data = self._adapter.read_from(addr, n_in)
        in_buf[in_start:in_end] = in_data

    def scan(self):
        """Optional helper - scan for I2C devices."""
        found = []
        for addr in range(0x08, 0x78):
            try:
                self._adapter.read_from(addr, 1)
                found.append(addr)
            except Exception:
                pass
        return found

    def deinit(self):
        try:
            if hasattr(self._adapter, "_serial"):
                self._adapter._serial.close()
        except Exception:
            pass


# ===================================================================
#  IMU state and reader thread
# ===================================================================

class IMUState:
    def __init__(self):
        self.lock = threading.Lock()
        self.quat = (1.0, 0.0, 0.0, 0.0)        # (w, x, y, z)
        self.euler = (0.0, 0.0, 0.0)            # (heading, roll, pitch) deg
        self.calib = (0, 0, 0, 0)               # (sys, gyro, accel, mag)
        self.temperature = 0.0
        self.last_update_time = 0.0
        self.error = None
        self.read_count = 0
        self.error_count = 0


class IMUThread(threading.Thread):
    def __init__(self, port, state):
        super().__init__(daemon=True)
        self.port = port
        self.state = state
        self.running = True
        self.sensor = None
        self.i2c = None

    def stop(self):
        self.running = False

    def _safe_read(self, attr_name):
        """Read a BNO055 attribute, return None on transient errors."""
        try:
            return getattr(self.sensor, attr_name)
        except Exception:
            return None

    def run(self):
        # ---- I2C bus init ----
        try:
            self.i2c = PololuI2CShim(self.port)
        except Exception as e:
            with self.state.lock:
                self.state.error = f"Pololu adapter open failed: {e}"
            print(f"Pololu adapter open failed: {e}")
            return

        # ---- BNO055 driver init ----
        # The BNO055 sometimes NACKs the very first read after power-on.
        # Retry a few times with a small delay.
        print("Initializing BNO055...")
        last_err = None
        for attempt in range(5):
            try:
                self.sensor = adafruit_bno055.BNO055_I2C(self.i2c,
                                                        address=BNO055_ADDR)
                # Force a real read to confirm the chip is responsive
                _ = self.sensor.calibration_status
                break
            except Exception as e:
                last_err = e
                print(f"  init attempt {attempt+1}/5: {e}")
                time.sleep(0.3)
        else:
            with self.state.lock:
                self.state.error = f"BNO055 init failed: {last_err}"
            print(f"BNO055 init failed after 5 attempts: {last_err}")
            return

        print("BNO055 streaming.")

        # ---- Streaming loop ----
        interval = 1.0 / SAMPLE_HZ
        next_tick = time.time()
        consecutive_errors = 0

        while self.running:
            quat = self._safe_read("quaternion")
            euler = self._safe_read("euler")
            calib = self._safe_read("calibration_status")
            temp = self._safe_read("temperature")

            # Defensive validation: a quaternion's magnitude should be ~1.
            # Magnitudes far from 1 indicate corrupted reads.
            quat_ok = (
                quat is not None
                and all(v is not None for v in quat)
                and 0.5 < sum(v * v for v in quat) ** 0.5 < 1.5
            )
            euler_ok = euler is not None and all(v is not None for v in euler)

            if quat_ok or euler_ok:
                consecutive_errors = 0
                with self.state.lock:
                    if quat_ok:
                        self.state.quat = quat
                    if euler_ok:
                        self.state.euler = euler
                    if calib:
                        self.state.calib = calib
                    if temp is not None:
                        self.state.temperature = temp
                    self.state.last_update_time = time.time()
                    self.state.read_count += 1
                    self.state.error = None
            else:
                consecutive_errors += 1
                with self.state.lock:
                    self.state.error_count += 1
                    if consecutive_errors >= 10:
                        self.state.error = "persistent read failures"

            # Rate limit
            next_tick += interval
            sleep = next_tick - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.time()

        try:
            self.i2c.deinit()
        except Exception:
            pass


# ===================================================================
#  HTML page (Three.js 3D viewer)
# ===================================================================

INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BNO055 IMU Test</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; background: #0d1117;
               color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont,
                                            "Segoe UI", sans-serif; }
  #viewer { position: fixed; top: 0; left: 0; right: 340px; bottom: 0; }
  #panel { position: fixed; top: 0; right: 0; bottom: 0; width: 340px;
           background: #161b22; border-left: 1px solid #30363d;
           padding: 20px; overflow-y: auto; }
  h2 { margin: 0 0 12px; font-size: 16px; font-weight: 600; }
  h3 { margin: 24px 0 8px; font-size: 12px; font-weight: 600;
       color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
  .stat { display: flex; justify-content: space-between; padding: 6px 0;
          font-size: 13px; border-bottom: 1px solid #21262d; }
  .stat .k { color: #8b949e; }
  .stat .v { color: #e6edf3; font-family: ui-monospace, "SF Mono", monospace; }
  .calib-row { display: flex; align-items: center; padding: 6px 0;
               border-bottom: 1px solid #21262d; }
  .calib-row .k { flex: 1; color: #8b949e; font-size: 13px; }
  .calib-row .v { font-family: ui-monospace, monospace; font-size: 13px;
                  margin-right: 10px; }
  .dots { display: inline-block; }
  .dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%;
         margin-right: 3px; background: #30363d; }
  .dot.lit { background: #3fb950; }
  .dot.all-lit { background: #58a6ff; }
  .help { margin-top: 20px; padding: 12px; background: #21262d;
          border-radius: 6px; font-size: 12px; color: #8b949e; line-height: 1.6; }
  .help b { color: #e6edf3; }
  .help.done { background: #1a3324; color: #7ee787; }
  .help.done b { color: #7ee787; }
  #status { padding: 8px 12px; background: #21262d; border-radius: 6px;
            font-size: 12px; margin-bottom: 12px; }
  #status.error { background: #4a1f1f; color: #ffa198; }
  #status.ok { background: #1a3324; color: #7ee787; }
</style>
</head>
<body>

<div id="viewer"></div>

<div id="panel">
  <h2>BNO055 IMU Test</h2>
  <div id="status">Connecting...</div>

  <h3>Calibration</h3>
  <div class="calib-row">
    <span class="k">System</span><span class="v" id="vCalSys">0</span>
    <span class="dots" id="dotsSys"></span>
  </div>
  <div class="calib-row">
    <span class="k">Gyro</span><span class="v" id="vCalGyro">0</span>
    <span class="dots" id="dotsGyro"></span>
  </div>
  <div class="calib-row">
    <span class="k">Accel</span><span class="v" id="vCalAccel">0</span>
    <span class="dots" id="dotsAccel"></span>
  </div>
  <div class="calib-row">
    <span class="k">Magnetometer</span><span class="v" id="vCalMag">0</span>
    <span class="dots" id="dotsMag"></span>
  </div>

  <div id="calibHelp" class="help">
    <b>Calibration in progress.</b><br>
    For each sensor at 0/3, do the corresponding motion:<br>
    &bull; <b>Gyro:</b> hold sensor still on a flat surface<br>
    &bull; <b>Accel:</b> slowly tilt to 6 orientations (die roll)<br>
    &bull; <b>Mag:</b> rotate in a big figure-8 in the air<br>
    System reaches 3 when all others do.
  </div>

  <h3>Euler angles (deg)</h3>
  <div class="stat"><span class="k">Heading / Yaw</span><span class="v" id="vYaw">0.0</span></div>
  <div class="stat"><span class="k">Roll</span><span class="v" id="vRoll">0.0</span></div>
  <div class="stat"><span class="k">Pitch</span><span class="v" id="vPitch">0.0</span></div>

  <h3>Quaternion</h3>
  <div class="stat"><span class="k">w</span><span class="v" id="vQw">1.000</span></div>
  <div class="stat"><span class="k">x</span><span class="v" id="vQx">0.000</span></div>
  <div class="stat"><span class="k">y</span><span class="v" id="vQy">0.000</span></div>
  <div class="stat"><span class="k">z</span><span class="v" id="vQz">0.000</span></div>

  <h3>Misc</h3>
  <div class="stat"><span class="k">Temperature</span><span class="v" id="vTemp">- C</span></div>
  <div class="stat"><span class="k">Sample count</span><span class="v" id="vCount">0</span></div>
  <div class="stat"><span class="k">Error count</span><span class="v" id="vErrors">0</span></div>
  <div class="stat"><span class="k">Rate</span><span class="v" id="vRate">- Hz</span></div>
</div>

<script type="importmap">
{ "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
} }
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const viewer = document.getElementById('viewer');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.PerspectiveCamera(
  50, viewer.clientWidth / viewer.clientHeight, 0.01, 100);
// Use Z-up convention (same as robotics / SLAM viewer)
camera.up.set(0, 0, 1);
camera.position.set(2.5, -2.5, 2);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(viewer.clientWidth, viewer.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);
viewer.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.7);
dirLight.position.set(3, 5, 4);
scene.add(dirLight);

// Grid on the XY plane (z=0 = floor), since we're Z-up
const grid = new THREE.GridHelper(4, 8, 0x444444, 0x2a2a2a);
grid.rotation.x = Math.PI / 2;  // rotate from XZ to XY
grid.position.z = -0.5;
scene.add(grid);

const worldAxes = new THREE.AxesHelper(0.3);
worldAxes.position.set(-1.5, -1.5, -0.5);
scene.add(worldAxes);

// ----- BNO055 chip model -----
const chipGroup = new THREE.Group();
scene.add(chipGroup);

const pcbGeom = new THREE.BoxGeometry(1.35, 1.0, 0.08);
const pcbMat = new THREE.MeshStandardMaterial({
  color: 0x1a4a8a, roughness: 0.6, metalness: 0.1,
});
const pcb = new THREE.Mesh(pcbGeom, pcbMat);
chipGroup.add(pcb);

const chipGeom = new THREE.BoxGeometry(0.28, 0.28, 0.08);
const chipMat = new THREE.MeshStandardMaterial({
  color: 0x111111, roughness: 0.4, metalness: 0.3,
});
const chip = new THREE.Mesh(chipGeom, chipMat);
chip.position.set(-0.15, 0.0, 0.08);
chipGroup.add(chip);

const dotGeom = new THREE.CircleGeometry(0.03, 16);
const dotMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
const dot = new THREE.Mesh(dotGeom, dotMat);
dot.position.set(-0.27, 0.12, 0.125);
chipGroup.add(dot);

for (let i = 0; i < 7; i++) {
  const pinGeom = new THREE.BoxGeometry(0.08, 0.08, 0.16);
  const pinMat = new THREE.MeshStandardMaterial({
    color: 0xc0a060, roughness: 0.3, metalness: 0.8,
  });
  const pin = new THREE.Mesh(pinGeom, pinMat);
  pin.position.set(0.5, 0.4 - i * 0.13, 0);
  chipGroup.add(pin);
}

const localAxes = new THREE.AxesHelper(1.0);
chipGroup.add(localAxes);

function makeAxisLabel(text, color) {
  const canvas = document.createElement('canvas');
  canvas.width = 64; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = color;
  ctx.font = 'bold 48px sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, 32, 32);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.3, 0.3, 1);
  return sprite;
}
const labelX = makeAxisLabel('X', '#ff5555'); labelX.position.set(1.1, 0, 0); chipGroup.add(labelX);
const labelY = makeAxisLabel('Y', '#55ff55'); labelY.position.set(0, 1.1, 0); chipGroup.add(labelY);
const labelZ = makeAxisLabel('Z', '#5599ff'); labelZ.position.set(0, 0, 1.1); chipGroup.add(labelZ);

// Apply BNO055 quaternion (w, x, y, z) to the chip group.
// Three.js Quaternion constructor takes (x, y, z, w).
function setQuaternion(w, x, y, z) {
  chipGroup.quaternion.set(x, y, z, w);
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

// ----- WebSocket + UI -----
const status = document.getElementById('status');
let messagesIn = 0;
let lastRateCheck = Date.now();
let lastRateCount = 0;

function setDots(elementId, n) {
  const container = document.getElementById(elementId);
  container.innerHTML = '';
  for (let i = 0; i < 3; i++) {
    const d = document.createElement('span');
    d.className = 'dot';
    if (i < n) d.classList.add(n === 3 ? 'all-lit' : 'lit');
    container.appendChild(d);
  }
}

function updateCalibHelp(sys, gyro, accel, mag) {
  const help = document.getElementById('calibHelp');
  if (sys === 3 && gyro === 3 && accel === 3 && mag === 3) {
    help.className = 'help done';
    help.innerHTML = '<b>Fully calibrated.</b> The sensor is ready.';
    return;
  }
  const lines = ['<b>Calibration in progress.</b><br>'];
  if (gyro < 3) lines.push('&bull; <b>Gyro:</b> hold sensor still on a flat surface<br>');
  if (accel < 3) lines.push('&bull; <b>Accel:</b> slowly tilt to 6 orientations (die roll)<br>');
  if (mag < 3) lines.push('&bull; <b>Mag:</b> rotate in a big figure-8 in the air<br>');
  if (sys < 3 && gyro === 3 && accel === 3 && mag === 3) {
    lines.push('&bull; System should follow shortly...');
  }
  help.className = 'help';
  help.innerHTML = lines.join('');
}

const wsUrl = `ws://${location.hostname}:%%WS_PORT%%`;
let ws;
function connect() {
  ws = new WebSocket(wsUrl);
  ws.onopen = () => { status.textContent = 'Connected'; status.className = 'ok'; };
  ws.onclose = () => {
    status.textContent = 'Disconnected (retrying)'; status.className = 'error';
    setTimeout(connect, 1000);
  };
  ws.onerror = () => {};
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.error) {
      status.textContent = 'Error: ' + m.error;
      status.className = 'error';
      return;
    }
    status.textContent = 'Streaming';
    status.className = 'ok';

    messagesIn++;
    setQuaternion(m.qw, m.qx, m.qy, m.qz);

    document.getElementById('vYaw').textContent   = m.yaw.toFixed(1);
    document.getElementById('vRoll').textContent  = m.roll.toFixed(1);
    document.getElementById('vPitch').textContent = m.pitch.toFixed(1);
    document.getElementById('vQw').textContent = m.qw.toFixed(3);
    document.getElementById('vQx').textContent = m.qx.toFixed(3);
    document.getElementById('vQy').textContent = m.qy.toFixed(3);
    document.getElementById('vQz').textContent = m.qz.toFixed(3);
    document.getElementById('vTemp').textContent = m.temp.toFixed(1) + ' C';
    document.getElementById('vCount').textContent = m.count;
    document.getElementById('vErrors').textContent = m.errors;

    const [sys, g, a, mg] = m.calib;
    document.getElementById('vCalSys').textContent   = sys;
    document.getElementById('vCalGyro').textContent  = g;
    document.getElementById('vCalAccel').textContent = a;
    document.getElementById('vCalMag').textContent   = mg;
    setDots('dotsSys', sys);
    setDots('dotsGyro', g);
    setDots('dotsAccel', a);
    setDots('dotsMag', mg);
    updateCalibHelp(sys, g, a, mg);

    const now = Date.now();
    if (now - lastRateCheck > 1000) {
      const rate = (messagesIn - lastRateCount) * 1000 / (now - lastRateCheck);
      document.getElementById('vRate').textContent = rate.toFixed(0) + ' Hz';
      lastRateCount = messagesIn;
      lastRateCheck = now;
    }
  };
}
connect();
</script>
</body>
</html>
"""


# ===================================================================
#  HTTP server
# ===================================================================

class IndexHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
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
        return


def serve_http():
    with socketserver.ThreadingTCPServer(("", HTTP_PORT), IndexHandler) as httpd:
        httpd.allow_reuse_address = True
        httpd.serve_forever()


# ===================================================================
#  WebSocket server
# ===================================================================

clients = set()


async def ws_handler(websocket):
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.discard(websocket)


async def broadcast_loop(state):
    interval = 1.0 / BROADCAST_HZ
    while True:
        await asyncio.sleep(interval)
        if not clients:
            continue

        with state.lock:
            w, x, y, z = state.quat
            yaw, roll, pitch = state.euler
            calib = state.calib
            temp = state.temperature
            count = state.read_count
            errors = state.error_count
            error = state.error

        msg = {
            "qw": w, "qx": x, "qy": y, "qz": z,
            "yaw": yaw, "roll": roll, "pitch": pitch,
            "calib": list(calib),
            "temp": temp,
            "count": count,
            "errors": errors,
            "error": error,
        }
        payload = json.dumps(msg)
        dead = []
        for ws in clients:
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        for d in dead:
            clients.discard(d)


async def run_websocket_server(state):
    async with websockets.serve(ws_handler, "", WS_PORT):
        await broadcast_loop(state)


# ===================================================================
#  Main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(description="BNO055 IMU test (via Pololu adapter)")
    p.add_argument("--port", default=DEFAULT_PORT,
                   help=f"Serial port of the Pololu adapter "
                        f"(default {DEFAULT_PORT})")
    p.add_argument("--no-browser", action="store_true",
                   help="Don't auto-open the browser")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Starting BNO055 IMU test viewer")
    print(f"  Pololu adapter on: {args.port}")

    state = IMUState()
    imu_thread = IMUThread(args.port, state)
    imu_thread.start()

    # Wait for the IMU thread to either succeed or fail
    print("Initializing BNO055 (may take a few seconds)...")
    t0 = time.time()
    while time.time() - t0 < 6.0:
        time.sleep(0.2)
        with state.lock:
            if state.error:
                print(f"\nError: {state.error}")
                sys.exit(1)
            if state.read_count > 0:
                break

    http_thread = threading.Thread(target=serve_http, daemon=True)
    http_thread.start()

    url = f"http://localhost:{HTTP_PORT}"
    print(f"\n  >> Open {url} in your browser  <<\n")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    print("Press Ctrl+C to stop.\n")

    try:
        asyncio.run(run_websocket_server(state))
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        imu_thread.stop()
        time.sleep(0.2)


if __name__ == "__main__":
    main()