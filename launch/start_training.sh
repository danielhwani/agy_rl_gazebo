#!/usr/bin/env bash
# Complete one-click runner for Gazebo TurtleBot3 Reinforcement Learning Training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source ROS 2 Humble & TurtleBot3 workspace
source /opt/ros/humble/setup.bash
if [ -f "/home/daniel/turtlebot3_ws/install/setup.bash" ]; then
    source /home/daniel/turtlebot3_ws/install/setup.bash
fi

export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/home/daniel/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models
export PYTHONPATH="${WORKSPACE_DIR}:${PYTHONPATH}"

# Check if Gazebo is already running
if ! pgrep -f "gzserver" > /dev/null; then
    echo "============================================================"
    echo "🤖 Starting Gazebo Simulation in Background..."
    echo "============================================================"
    ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage1.launch.py &
    GAZEBO_PID=$!
    
    # Wait for ROS 2 topics to be ready
    echo "[*] Waiting for Gazebo /scan & /odom topics..."
    for i in {1..30}; do
        if ros2 topic list 2>/dev/null | grep -q "/scan"; then
            echo "[✓] Gazebo simulation is ready!"
            break
        fi
        sleep 1
    done
else
    echo "[*] Existing Gazebo instance detected. Connecting directly..."
fi

# Run training in rl_gazebo conda environment with system libstdc++ compatibility
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
PYTHON_BIN="/home/daniel/miniconda3/envs/rl_gazebo/bin/python"

cd "${WORKSPACE_DIR}"
echo "============================================================"
echo "🎯 Running PyTorch DQN Training Node..."
echo "============================================================"
"${PYTHON_BIN}" -m gazebo_rl.train "$@"
