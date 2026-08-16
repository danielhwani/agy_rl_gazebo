#!/usr/bin/env python3
"""
Training Script for Gazebo TurtleBot3 Reinforcement Learning (DQN).
Handles the main training loop, metric logging, model checkpointing, and live plot updates.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt

import rclpy
from gazebo_rl.gazebo_env import GazeboTurtleBotEnv
from gazebo_rl.dqn_agent import DQNAgent


def plot_training_results(history: dict, save_path: str):
    """Generates and saves training metric plots."""
    episodes = list(range(1, len(history["rewards"]) + 1))
    if len(episodes) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("TurtleBot3 Gazebo RL (DQN) Training Progress", fontsize=14, fontweight="bold")

    # 1. Episode Rewards & Rolling Average
    ax1 = axes[0, 0]
    ax1.plot(episodes, history["rewards"], alpha=0.35, color="blue", label="Episode Reward")
    if len(history["rewards"]) >= 5:
        rolling_mean = np.convolve(history["rewards"], np.ones(5) / 5, mode="valid")
        ax1.plot(range(5, len(history["rewards"]) + 1), rolling_mean, color="darkblue", linewidth=2, label="5-Ep MA")
    ax1.set_title("Reward Progression")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # 2. Episode Steps
    ax2 = axes[0, 1]
    ax2.plot(episodes, history["steps"], color="orange", linewidth=1.5)
    ax2.set_title("Steps per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. Epsilon Decay
    ax3 = axes[1, 0]
    ax3.plot(episodes, history["epsilons"], color="green", linewidth=1.5)
    ax3.set_title("Exploration Rate (Epsilon)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Epsilon")
    ax3.grid(True, linestyle="--", alpha=0.6)

    # 4. Success / Collision Summary
    ax4 = axes[1, 1]
    statuses = history.get("statuses", [])
    goals = sum(1 for s in statuses if s == "goal_reached")
    collisions = sum(1 for s in statuses if s == "collision")
    timeouts = sum(1 for s in statuses if s == "timeout")
    bars = ax4.bar(["Goal Reached", "Collision", "Timeout"], [goals, collisions, timeouts],
                   color=["#2ecc71", "#e74c3c", "#f39c12"])
    ax4.set_title(f"Outcomes (Total: {len(statuses)})")
    ax4.set_ylabel("Count")
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f"{height}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha="center", va="bottom", fontweight="bold")
    ax4.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train TurtleBot3 DQN in Gazebo")
    parser.add_argument("--episodes", type=int, default=150, help="Total training episodes")
    parser.add_argument("--max-steps", type=int, default=400, help="Max steps per episode")
    parser.add_argument("--save-freq", type=int, default=10, help="Model saving frequency (episodes)")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save PyTorch models")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to save logs and plots")
    parser.add_argument("--load-model", type=str, default=None, help="Path to checkpoint model to resume")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_json_path = os.path.join(args.log_dir, f"training_log_{timestamp}.json")
    plot_png_path = os.path.join(args.log_dir, f"training_plot_{timestamp}.png")
    best_model_path = os.path.join(args.model_dir, "dqn_stage1_best.pth")
    last_model_path = os.path.join(args.model_dir, "dqn_stage1_latest.pth")

    rclpy.init()
    env = GazeboTurtleBotEnv()
    env.max_step_per_episode = args.max_steps

    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=args.lr
    )

    if args.load_model and os.path.exists(args.load_model):
        print(f"[*] Resuming training from checkpoint: {args.load_model}")
        agent.load(args.load_model)

    print("\n" + "=" * 60)
    print("🚀 Starting Gazebo TurtleBot3 Reinforcement Learning (DQN)")
    print(f"• Total Episodes: {args.episodes}")
    print(f"• Max Steps/Ep : {args.max_steps}")
    print(f"• State Dim    : {env.state_dim} (24 LiDAR + Goal Dist + Heading Error)")
    print(f"• Action Dim   : {env.action_dim} (Discrete CmdVel actions)")
    print("=" * 60 + "\n")

    history = {
        "rewards": [],
        "steps": [],
        "epsilons": [],
        "losses": [],
        "statuses": []
    }

    best_reward = -float("inf")
    start_total_time = time.time()

    try:
        for ep in range(1, args.episodes + 1):
            ep_start_time = time.time()
            state = env.reset(random_goal=True)
            ep_reward = 0.0
            ep_loss = []
            done = False
            step = 0
            final_status = "unknown"

            while not done and step < args.max_steps:
                step += 1
                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)

                agent.memory.push(state, action, reward, next_state, done)
                loss = agent.update()
                if loss is not None:
                    ep_loss.append(loss)

                ep_reward += reward
                state = next_state
                final_status = info.get("status", "unknown")

            agent.decay_epsilon()
            ep_duration = time.time() - ep_start_time
            avg_loss = np.mean(ep_loss) if len(ep_loss) > 0 else 0.0

            history["rewards"].append(float(ep_reward))
            history["steps"].append(int(step))
            history["epsilons"].append(float(agent.epsilon))
            history["losses"].append(float(avg_loss))
            history["statuses"].append(final_status)

            status_icon = "🎯 [GOAL]" if final_status == "goal_reached" else ("💥 [COLL]" if final_status == "collision" else "⏳ [TIME]")
            print(f"Ep {ep:03d}/{args.episodes:03d} | {status_icon} | Reward: {ep_reward:7.1f} | Steps: {step:3d} | "
                  f"Eps: {agent.epsilon:.3f} | Loss: {avg_loss:.4f} | Time: {ep_duration:.1f}s")

            # Check for new best model
            recent_avg = np.mean(history["rewards"][-10:])
            if recent_avg > best_reward and ep >= 10:
                best_reward = recent_avg
                agent.save(best_model_path)
                print(f"    ⭐ New Best 10-Ep Avg Reward: {best_reward:.1f} -> Saved to {best_model_path}")

            # Periodic saving & plot generation
            if ep % args.save_freq == 0 or ep == args.episodes:
                agent.save(last_model_path)
                with open(log_json_path, "w") as f:
                    json.dump(history, f, indent=2)
                plot_training_results(history, plot_png_path)
                print(f"    📊 Saved checkpoint & progress plot ({plot_png_path})")

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user. Saving current weights...")
        agent.save(last_model_path)
        with open(log_json_path, "w") as f:
            json.dump(history, f, indent=2)
        plot_training_results(history, plot_png_path)

    finally:
        total_time = time.time() - start_total_time
        print("\n" + "=" * 60)
        print(f"🏁 Training Complete! (Total Duration: {total_time/60:.2f} mins)")
        print(f"• Best Model : {best_model_path}")
        print(f"• Latest Model: {last_model_path}")
        print(f"• Plot Result : {plot_png_path}")
        print("=" * 60)
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
