#!/usr/bin/bash

conda activate g1-cartesian
conda install -c conda-forge fastapi uvicorn pyyaml

export XR_TELEOPERATE_ROOT=/home/g1gen5/Projects/robo/.third_party/src/xr_teleoperate
export UNITREE_SDK2PY_ROOT=/home/g1gen5/Projects/robo/.third_party/src/unitree_sdk2_python
export ROBOT_NETWORK_INTERFACE=enP2p1s0