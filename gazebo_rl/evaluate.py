#!/usr/bin/env python3
"""
Evaluation script for trained TurtleBot3 DQN Agent in Gazebo.
Runs test episodes with pure greedy policy (epsilon = 0) and reports performance metrics.
"""

import os
import time
import argparse
import numpy as np

import rclpy
from gazebo_rl.gazebo_env import GazeboTurtleBotEnv
from gazebo_rl.dqn_agent import DQNAgent


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained TurtleBot3 DQN agent in Gazebo")
    parser.add_argument("--model-path", type=str, default="models/dqn_stage1_best.pth",
                        help="Path to trained PyTorch model (.pth)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--max-steps", type=int, default=400, help="Max steps per episode")
    args = parser.parse_args()

    rclpy.init()
    env = GazeboTurtleBotEnv(node_name="gazebo_tb3_evaluator")
    env.max_step_per_episode = args.max_steps

    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim
    )

    if not os.path.exists(args.model_path):
        # Check latest if best doesn't exist
        alt_path = "models/dqn_stage1_latest.pth"
        if os.path.exists(alt_path):
            args.model_path = alt_path
        else:
            print(f"[!] Model path '{args.model_path}' not found. Please train a model first with train.py.")
            return

    print(f"[*] Loading trained model from: {args.model_path}")
    agent.load(args.model_path)

    print("\n" + "=" * 60)
    print("🤖 Starting TurtleBot3 Gazebo RL Evaluation (Greedy Policy)")
    print(f"• Test Episodes: {args.episodes}")
    print(f"• Model Path   : {args.model_path}")
    print("=" * 60 + "\n")

    results = []
    success_count = 0

    try:
        for ep in range(1, args.episodes + 1):
            state = env.reset(random_goal=True)
            ep_reward = 0.0
            step = 0
            done = False
            status = "unknown"

            while not done and step < args.max_steps:
                step += 1
                action = agent.get_action(state, evaluate=True)  # Greedy action
                next_state, reward, done, info = env.step(action)
                ep_reward += reward
                state = next_state
                status = info.get("status", "unknown")

            if status == "goal_reached":
                success_count += 1
                status_icon = "🎯 [SUCCESS]"
            elif status == "collision":
                status_icon = "💥 [COLLISION]"
            else:
                status_icon = "⏳ [TIMEOUT]"

            print(f"Test Ep {ep:02d}/{args.episodes:02d} | {status_icon} | Reward: {ep_reward:7.1f} | Steps: {step:3d}")
            results.append({"reward": ep_reward, "steps": step, "status": status})
            time.sleep(0.5)

    finally:
        success_rate = (success_count / args.episodes) * 100.0 if args.episodes > 0 else 0
        avg_reward = np.mean([r["reward"] for r in results]) if results else 0
        avg_steps = np.mean([r["steps"] for r in results if r["status"] == "goal_reached"]) if success_count > 0 else 0

        print("\n" + "=" * 60)
        print("📊 Evaluation Summary:")
        print(f"• Success Rate  : {success_rate:.1f}% ({success_count}/{args.episodes})")
        print(f"• Average Reward: {avg_reward:.2f}")
        if success_count > 0:
            print(f"• Avg Steps/Goal: {avg_steps:.1f} steps")
        print("=" * 60)

        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
