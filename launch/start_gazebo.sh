#!/usr/bin/env bash
# Script to launch TurtleBot3 Gazebo Stage 1 World

set -e

# Source ROS 2 Humble & TurtleBot3 workspace
source /opt/ros/humble/setup.bash
if [ -f "/home/daniel/turtlebot3_ws/install/setup.bash" ]; then
    source /home/daniel/turtlebot3_ws/install/setup.bash
fi

export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/home/daniel/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models

echo "============================================================"
echo "🤖 Launching TurtleBot3 Gazebo (Stage 1 World)..."
echo "• Model: $TURTLEBOT3_MODEL"
echo "• Display: $DISPLAY"
echo "============================================================"

ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage1.launch.py
