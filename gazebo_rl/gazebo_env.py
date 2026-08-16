#!/usr/bin/env python3
"""
Gazebo TurtleBot3 Reinforcement Learning Environment
Provides a pure Python / ROS 2 interface without any Gymnasium dependency.
"""

import math
import time
import random
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from std_srvs.srv import Empty

# Minimal SDF for visual target goal marker in Gazebo
GOAL_SDF = """<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='goal_box'>
    <pose frame=''>0 0 0 0 0 0</pose>
    <link name='goal_link'>
      <visual name='goal_visual'>
        <pose frame=''>0 0 0.01 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>0.22</radius>
            <length>0.02</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.9 0.1 0.1 0.9</ambient>
          <diffuse>0.9 0.1 0.1 0.9</diffuse>
          <emissive>0.4 0.0 0.0 1.0</emissive>
        </material>
      </visual>
    </link>
    <static>1</static>
  </model>
</sdf>"""


class GazeboTurtleBotEnv(Node):
    """
    Gazebo Reinforcement Learning Environment for TurtleBot3.
    """

    def __init__(self, node_name: str = "gazebo_tb3_rl_env", num_scan_samples: int = 24):
        super().__init__(node_name)

        self.num_scan_samples = num_scan_samples
        self.max_scan_range = 3.5    # Max laser range (meters)
        self.collision_dist = 0.13    # Collision threshold (meters) for Burger
        self.goal_reach_dist = 0.25   # Goal arrival threshold (meters)
        self.max_step_per_episode = 400

        # Robot state tracking
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.latest_scan = None
        self.odom_received = False
        self.scan_received = False

        # Goal tracking
        self.goal_x = 1.5
        self.goal_y = 1.5
        self.prev_distance_to_goal = 0.0
        self.goal_spawned = False
        self.step_count = 0

        # Action space definition: (linear_velocity m/s, angular_velocity rad/s)
        self.action_space = [
            (0.18, 0.0),    # 0: Forward
            (0.06, 0.8),    # 1: Left Turn
            (0.06, -0.8),   # 2: Right Turn
            (0.14, 0.35),   # 3: Soft Left
            (0.14, -0.35),  # 4: Soft Right
        ]
        self.action_dim = len(self.action_space)
        self.state_dim = self.num_scan_samples + 2  # 24 LiDAR sectors + (distance, heading_angle)

        # Pre-defined goal positions in Stage 1 arena (5m x 5m)
        self.goal_candidates = [
            (1.5, 1.5), (-1.5, 1.5), (1.5, -1.5), (-1.5, -1.5),
            (1.7, 0.0), (-1.7, 0.0), (0.0, 1.7), (0.0, -1.7),
            (1.2, 1.0), (-1.2, 1.0), (1.0, -1.2), (-1.0, -1.2)
        ]

        # Setup ROS2 Publishers & Subscribers
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self._scan_callback, qos)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_callback, 10)

        # Setup Gazebo Service Clients
        self.reset_sim_client = self.create_client(Empty, "/reset_simulation")
        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete_client = self.create_client(DeleteEntity, "/delete_entity")

        self.get_logger().info(f"GazeboTurtleBotEnv initialized (State Dim: {self.state_dim}, Action Dim: {self.action_dim})")

    def _scan_callback(self, msg: LaserScan):
        self.latest_scan = msg.ranges
        self.scan_received = True

    def _odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # Convert quaternion to yaw angle
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.odom_received = True

    def _spin_once(self, timeout_sec: float = 0.05):
        """Processes incoming ROS2 messages."""
        rclpy.spin_once(self, timeout_sec=timeout_sec)

    def _wait_for_sensor_data(self, timeout: float = 3.0) -> bool:
        """Wait until fresh scan and odom data are available."""
        self.scan_received = False
        self.odom_received = False
        start_time = time.time()
        while time.time() - start_time < timeout:
            self._spin_once(0.02)
            if self.scan_received and self.odom_received:
                return True
        return False

    def _get_processed_scan(self) -> np.ndarray:
        """
        Processes 360-degree LiDAR into num_scan_samples sectors.
        Handles inf/nan and normalizes to [0, 1].
        """
        if self.latest_scan is None or len(self.latest_scan) == 0:
            return np.ones(self.num_scan_samples, dtype=np.float32)

        ranges = np.array(self.latest_scan, dtype=np.float32)
        # Replace inf, nan, and out-of-range values
        ranges = np.nan_to_num(ranges, nan=self.max_scan_range, posinf=self.max_scan_range, neginf=self.max_scan_range)
        ranges = np.clip(ranges, 0.05, self.max_scan_range)

        # Downsample into sectors (taking min range in each sector for obstacle safety)
        sector_size = len(ranges) // self.num_scan_samples
        sampled_ranges = []
        for i in range(self.num_scan_samples):
            start_idx = i * sector_size
            end_idx = (i + 1) * sector_size
            sector_min = np.min(ranges[start_idx:end_idx])
            sampled_ranges.append(sector_min / self.max_scan_range)

        return np.array(sampled_ranges, dtype=np.float32)

    def _get_goal_features(self) -> tuple[float, float, float]:
        """
        Calculates distance and relative heading error to the goal.
        Returns: (distance, heading_error, heading_angle)
        """
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        distance = math.sqrt(dx * dx + dy * dy)

        target_angle = math.atan2(dy, dx)
        heading_error = target_angle - self.robot_yaw

        # Normalize heading error to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2.0 * math.pi
        while heading_error < -math.pi:
            heading_error += 2.0 * math.pi

        return distance, heading_error, target_angle

    def get_state(self) -> np.ndarray:
        """
        Builds the normalized state vector (Dim: num_scan_samples + 2):
        [LiDAR sectors (0~1), normalized goal distance (0~1), normalized heading error (-1~1)]
        """
        scan_feats = self._get_processed_scan()
        distance, heading_error, _ = self._get_goal_features()

        # Normalized distance (max arena span ~5m)
        norm_dist = np.clip(distance / 5.0, 0.0, 1.0)
        # Normalized heading error [-1.0, 1.0]
        norm_heading = heading_error / math.pi

        state = np.hstack([scan_feats, [norm_dist, norm_heading]]).astype(np.float32)
        return state

    def respawn_goal(self, x: float, y: float):
        """Updates the target goal location and moves the visual marker in Gazebo."""
        self.goal_x = float(x)
        self.goal_y = float(y)

        # Delete previous goal marker if spawned
        if self.goal_spawned and self.delete_client.service_is_ready():
            req_del = DeleteEntity.Request()
            req_del.name = "goal_box"
            future_del = self.delete_client.call_async(req_del)
            rclpy.spin_until_future_complete(self, future_del, timeout_sec=0.2)

        # Spawn goal entity at new location
        if self.spawn_client.service_is_ready():
            req_spawn = SpawnEntity.Request()
            req_spawn.name = "goal_box"
            req_spawn.xml = GOAL_SDF
            req_spawn.initial_pose.position.x = self.goal_x
            req_spawn.initial_pose.position.y = self.goal_y
            req_spawn.initial_pose.position.z = 0.01
            future_spawn = self.spawn_client.call_async(req_spawn)
            rclpy.spin_until_future_complete(self, future_spawn, timeout_sec=0.2)
            self.goal_spawned = True

    def reset_simulation_world(self):
        """Resets the Gazebo world and robot back to the starting point."""
        self.publish_cmd_vel(0.0, 0.0)

        if self.reset_sim_client.service_is_ready():
            req = Empty.Request()
            future = self.reset_sim_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)

    def publish_cmd_vel(self, linear: float, angular: float):
        """Publishes velocity command to /cmd_vel."""
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_vel_pub.publish(msg)

    def reset(self, random_goal: bool = True) -> np.ndarray:
        """
        Resets environment for a new episode.
        Returns initial state observation.
        """
        self.step_count = 0

        # Reset robot & world in Gazebo
        self.reset_simulation_world()

        # Choose goal position
        if random_goal:
            gx, gy = random.choice(self.goal_candidates)
        else:
            gx, gy = 1.5, 1.5
        self.respawn_goal(gx, gy)

        # Allow simulation physics to settle and read fresh sensor data
        time.sleep(0.1)
        self._wait_for_sensor_data()

        dist, _, _ = self._get_goal_features()
        self.prev_distance_to_goal = dist

        return self.get_state()

    def step(self, action_idx: int, step_duration: float = 0.1) -> tuple[np.ndarray, float, bool, dict]:
        """
        Executes one RL environment step:
        1. Apply velocity command
        2. Wait step_duration while spinning ROS2 callbacks
        3. Compute next state, reward, collision/goal status
        """
        self.step_count += 1
        linear_v, angular_v = self.action_space[action_idx]
        self.publish_cmd_vel(linear_v, angular_v)

        # Step simulation time
        start_time = time.time()
        while time.time() - start_time < step_duration:
            self._spin_once(0.01)

        # Compute next state
        next_state = self.get_state()
        curr_dist, heading_error, _ = self._get_goal_features()

        # Check LiDAR minimum distance for collision
        scan_ranges = np.nan_to_num(
            np.array(self.latest_scan) if self.latest_scan else np.array([self.max_scan_range]),
            nan=self.max_scan_range, posinf=self.max_scan_range
        )
        min_scan = float(np.min(scan_ranges)) if len(scan_ranges) > 0 else self.max_scan_range

        # Reward Calculation
        # Progress reward (positive when moving closer to goal)
        dist_progress = (self.prev_distance_to_goal - curr_dist)
        reward = dist_progress * 50.0

        # Small step cost to encourage fast navigation
        reward -= 0.05

        # Small penalty for facing away from goal
        reward -= (abs(heading_error) / math.pi) * 0.15

        self.prev_distance_to_goal = curr_dist
        done = False
        info = {"status": "in_progress", "distance": curr_dist, "step": self.step_count}

        # Check Termination Conditions
        if curr_dist < self.goal_reach_dist:
            # Reached Goal!
            reward = 200.0
            done = True
            info["status"] = "goal_reached"
            self.publish_cmd_vel(0.0, 0.0)
            self.get_logger().info(f"🎯 Goal Reached in {self.step_count} steps! (Reward: +200)")

        elif min_scan < self.collision_dist:
            # Collision with obstacle or wall
            reward = -100.0
            done = True
            info["status"] = "collision"
            self.publish_cmd_vel(0.0, 0.0)
            self.get_logger().warn(f"💥 Collision detected at {min_scan:.2f}m! (Reward: -100)")

        elif self.step_count >= self.max_step_per_episode:
            # Time limit exceeded
            reward = -20.0
            done = True
            info["status"] = "timeout"
            self.publish_cmd_vel(0.0, 0.0)
            self.get_logger().info(f"⏳ Episode timeout ({self.step_count} steps).")

        return next_state, reward, done, info

    def close(self):
        """Stops robot and cleans up environment."""
        self.publish_cmd_vel(0.0, 0.0)
        self.destroy_node()
