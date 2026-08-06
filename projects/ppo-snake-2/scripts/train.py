"""ppo-snake-2 PPO 训练入口。

两种模式：
  1. 纯 PPO：随机初始化直接强化学习
  2. BC+PPO：先跑 BC 预训练（或加载已有 bc_policy.json），再用 PPO 微调

用法：
  python scripts/train.py --iters 600 --tag s1
  python scripts/train.py --iters 600 --tag s2 --bc bc_policy.json
"""

import argparse
import csv
import os
import random
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.snake_game import STATE_DIM
from src.snake_vec import VectorSnakeEnv
from src.ppo import BottleneckPolicy, PPOTrainer


def parse_args():
    p = argparse.ArgumentParser(description="PPO 训练贪吃蛇")
    p.add_argument("--iters", type=int, default=600, help="PPO 迭代轮数")
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=1024,
                   help="每迭代 rollout 步数")
    p.add_argument("--epochs", type=int, default=4, help="PPO 内层轮数")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=56)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step-penalty", type=float, default=0.0,
                   help="每步惩罚（负值，激励更快吃食物）")
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--kl-threshold", type=float, default=0.05)
    p.add_argument("--bc", type=str, default=None,
                   help="BC 预训练权重路径（可选）")
    p.add_argument("--tag", type=str, default="s1")
    p.add_argument("--eval-every", type=int, default=20,
                   help="每 N 迭代评估一次并保存 best")
    return p.parse_args()


def main():
    args = parse_args()
    # 固定全局 RNG（网络初始化、rollout 采样、minibatch shuffle 均受 seed 控制）
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = os.path.join("experiments", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    net = BottleneckPolicy(STATE_DIM, 3, hidden=args.hidden,
                           n_blocks=args.blocks, seed=args.seed)
    if args.bc:
        net.load(args.bc)
        print(f"[train] 加载 BC 起点: {args.bc}")
    else:
        print("[train] 纯 PPO（随机初始化）")

    trainer = PPOTrainer(net, lr=args.lr, ent_coef=args.ent_coef,
                         kl_threshold=args.kl_threshold)
    envs = VectorSnakeEnv(n_envs=args.n_envs,
                          seeds=list(range(args.seed,
                                           args.seed + args.n_envs)))

    def step_reward(buf):
        if args.step_penalty == 0.0:
            return buf
        # 在非吃食物/非死亡步叠加惩罚
        mask = (buf["rewards"] == 0.0)
        buf["rewards"] = buf["rewards"] + mask * args.step_penalty
        return buf

    log_path = os.path.join(out_dir, "train_log.csv")
    best_path = os.path.join(out_dir, "best_snake.json")
    best_len = 0.0
    logf = open(log_path, "w", newline="")
    writer = csv.writer(logf)
    writer.writerow(["iter", "avg_reward", "avg_food", "avg_length",
                     "clip_frac", "approx_kl"])

    t0 = time.time()
    for it in range(1, args.iters + 1):
        rollout = trainer.collect_rollout(envs, max_steps=args.max_steps)
        rollout = step_reward(rollout)
        pl, vl, ent, kl, clip_frac = trainer.update(
            rollout, epochs=args.epochs, minibatch=512)

        if it % args.eval_every == 0 or it == 1:
            avg_len, avg_food = trainer.evaluate(n_episodes=32)
            # 更新 best（保存 avg_length 最高的模型）
            if avg_len > best_len:
                best_len = avg_len
                net.save(best_path)
            avg_reward = float(rollout["rewards"].mean())
            writer.writerow([it, f"{avg_reward:.1f}", f"{avg_food:.1f}",
                             f"{avg_len:.1f}",
                             f"{clip_frac:.3f}", f"{kl:.4f}"])
            logf.flush()
            el = time.time() - t0
            print(f"[{it}/{args.iters}] len={avg_len:.1f} food={avg_food:.1f} "
                  f"clip={clip_frac:.3f} kl={kl:.4f} "
                  f"({el:.0f}s)")

    logf.close()
    print(f"[train] 完成。best 身长 {best_len:.1f} 保存于 {best_path}")


if __name__ == "__main__":
    main()
