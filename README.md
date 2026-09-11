# Classic Control

Local browser-based Cartesian control for a Unitree G1 with two Dex3 hands. It provides checked dual-arm IK, independent wrist targets, a fixed neutral waist action, hand tuning, and named grasp experiments without terminal teleoperation.

This project directly commands physical hardware. Support the robot, clear its workspace, keep its physical E-stop reachable, and ensure no other process publishes to `rt/lowcmd` or either Dex3 command topic.

## Requirements

- Python 3.10 or newer
- Unitree `unitree_sdk2py`, CycloneDDS, Pinocchio, and their native runtime dependencies
- A compatible `xr_teleoperate` checkout
- A G1-29 with Dex3 state topics available

The vendor stack is deliberately not declared as a portable PyPI dependency. Install this project inside the working robot environment:

```bash
python -m pip install -e .
export XR_TELEOPERATE_ROOT=/path/to/xr_teleoperate
export ROBOT_NETWORK_INTERFACE=enP2p1s0  # optional if CycloneDDS is already configured
classic-control
```

Open <http://127.0.0.1:8000>. The server intentionally binds only to localhost and always connects to real hardware.

`CLASSIC_CONTROL_GRASPS` may point to a different writable grasp YAML file. By default, the application uses [`config/grasps.yaml`](config/grasps.yaml).

## Controls and safety behavior

- Hold **Space** while requesting motion. Jog keys are `W/S`, `A/D`, and `R/F`; Shift makes a jog three times larger.
- Releasing Space cancels the active trajectory and continuously holds its most recent command.
- **Escape** or **Release robot control** cancels motion, stops this application's command publishers, and returns the G1 to AI mode. Losing the browser connection does the same.
- **Normal pose** returns both arms to the conservative neutral pose and sets waist yaw, roll, and pitch to zero. It does not alter the measured leg targets.
- Hand sliders stage values only. Holding Space and pressing Apply performs a bounded, interpolated move.
- There are no confirmation phrases or confirmation dialogs.

Only one browser may own the control WebSocket. IK failures, invalid feedback, limit violations, and command publisher failures are shown as faults and block motion until control is released.

## Grasps

Grasps use schema version 1 and named logical joints, so the left/right DDS ordering difference stays inside the hand driver. A grasp may define either or both hands. The required `open` grasp covers both hands; the initial `right-pinch` is calibrated only for the right hand. Tune the left hand from its measured state before saving left-hand grasps.

UI writes are range-validated and atomic. Saving a grasp updates the YAML file, so useful physical experiments can be reviewed and committed.

## Development

The package is split by responsibility:

- `arm.py` contains hardware-independent transforms plus the checked Unitree IK wrapper.
- `hardware.py` is the only owner of DDS command publishers.
- `service.py` serializes commands and implements deadman, hold, release, and fault behavior.
- `web.py` and `static/` implement the local API and interface.

Run the hardware-free suite with:

```bash
python -m unittest discover -s tests -v
python -m compileall -q classic_control
```

For physical acceptance, test in this order: state connection, acquire/release, hand open, one hand joint at a time, Normal pose, small single-arm translations, rotations, absolute targets, elbow biases, then dual-hand grasps.

## License

No license has been granted. All rights are reserved until a license is selected.
