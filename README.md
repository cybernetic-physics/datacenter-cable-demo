# Classic Control

Local browser-based Cartesian control for a Unitree G1 with two Dex3 hands. It provides checked dual-arm IK, independent wrist targets, a fixed neutral waist action, hand tuning, and named grasp experiments without terminal teleoperation.

This project directly commands physical hardware. Support the robot, clear its workspace, keep its physical E-stop reachable, and ensure no other process publishes to `rt/lowcmd` or either Dex3 command topic.

## Conda environment

Use Conda for this project. Pinocchio and the rest of the compiled robotics stack must come from the same Conda environment; do not build this environment with pip or uv.

To create a standalone environment:

```bash
conda env create -f environment.yml
conda activate classic-control

export XR_TELEOPERATE_ROOT=/path/to/xr_teleoperate
export UNITREE_SDK2PY_ROOT=/path/to/unitree_sdk2_python
export ROBOT_NETWORK_INTERFACE=enP2p1s0

python -m classic_control.web
```

`unitree_sdk2py` is loaded directly from its source checkout because it is not available from conda-forge. If it is already installed in the active Conda environment, `UNITREE_SDK2PY_ROOT` may be omitted.

To reuse the existing `g1-cartesian` environment instead, install only the missing UI packages through Conda:

```bash
conda activate g1-cartesian
conda install -c conda-forge fastapi uvicorn pyyaml pyzmq

export XR_TELEOPERATE_ROOT=/path/to/xr_teleoperate
export ROBOT_NETWORK_INTERFACE=enP2p1s0
python -m classic_control.web
```

Run these commands from the repository root; the project itself does not need to be installed with pip.

Open <http://127.0.0.1:8000>. The server intentionally binds only to localhost and always connects to real hardware.

The Robot view selects between the G1 PC2 D435i RGB feed, the taped side-by-side
head stereo feed, and an explicitly labelled Thor AIRHUG fallback. The two
robot-mounted feeds arrive as latest-value JPEG frames from Teleimager at
`192.168.123.164`; set `TELEIMAGER_HOST` to override that address. The Thor
fallback uses native JPEG through the system GStreamer installation; set
`CLASSIC_CONTROL_CAMERA_DEVICE` to override its automatic AIRHUG discovery.

### PC2 camera service

The PC2 camera configuration is tracked in
[`config/teleimager-pc2.yaml`](config/teleimager-pc2.yaml). It assigns the
internal D435i RGB camera to port `55555` and the taped head stereo pair to
port `55556`. Deploy it and start Teleimager on PC2 with:

```bash
scp config/teleimager-pc2.yaml unitree@192.168.123.164:/home/unitree/teleimager_pkg/cam_config_server.yaml
ssh unitree@192.168.123.164 'cd /home/unitree/teleimager_pkg && ./imgsrv_ctl.sh start'
```

Check or stop it with:

```bash
ssh unitree@192.168.123.164 'cd /home/unitree/teleimager_pkg && ./imgsrv_ctl.sh status'
ssh unitree@192.168.123.164 'cd /home/unitree/teleimager_pkg && ./imgsrv_ctl.sh stop'
```

Do not run `robonia-pc2-sensor-agent.service` and Teleimager together: both
open the same cameras. The sensor agent is enabled on PC2 by default, so after
a PC2 reboot stop it before starting Teleimager. Making that choice persistent
requires disabling the sensor-agent service with `sudo systemctl disable
robonia-pc2-sensor-agent.service` on PC2.

`CLASSIC_CONTROL_GRASPS` may point to a different writable grasp YAML file. By default, the application uses [`config/grasps.yaml`](config/grasps.yaml).

## Controls and safety behavior

- Click a motion control or use the `W/S`, `A/D`, and `R/F` jog keys; Shift makes a jog three times larger. No key needs to be held.
- **Stop motion** cancels the active trajectory and continuously holds its most recent command.
- **Escape** or **Release robot control** cancels motion, stops this application's command publishers, and returns the G1 to AI mode. Losing the browser connection does the same.
- **Normal pose** returns both arms to the conservative neutral pose and sets waist yaw, roll, and pitch to zero. It does not alter the measured leg targets.
- Hand sliders stage values only. Pressing Apply performs a bounded, interpolated move.
- There are no confirmation phrases or confirmation dialogs.

Only one browser may own the control WebSocket. IK failures, invalid feedback, limit violations, and command publisher failures are shown as faults and block motion until control is released.

## Grasps

Grasps use schema version 1 and named logical joints, so the left/right DDS ordering difference stays inside the hand driver. A grasp may define either or both hands. The required `open` grasp covers both hands; the initial `right-pinch` is calibrated only for the right hand. Tune the left hand from its measured state before saving left-hand grasps.

UI writes are range-validated and atomic. Saving a grasp updates the YAML file, so useful physical experiments can be reviewed and committed.

## Development

The package is split by responsibility:

- `arm.py` contains hardware-independent transforms plus the checked Unitree IK wrapper.
- `hardware.py` is the only owner of DDS command publishers.
- `service.py` serializes commands and implements stop/hold, release, and fault behavior.
- `camera.py` independently bridges the head camera to a browser MJPEG stream.
- `web.py` and `static/` implement the local API and interface.

Run the hardware-free suite from the activated Conda environment with:

```bash
python -m unittest discover -s tests -v
python -m compileall -q classic_control
```

For physical acceptance, test in this order: state connection, acquire/release, hand open, one hand joint at a time, Normal pose, small single-arm translations, rotations, absolute targets, elbow biases, then dual-hand grasps.

## License

No license has been granted. All rights are reserved until a license is selected.
