"""ppo-snake-2 BC 预训练：用贪心 Z 字专家数据行为克隆。

用法：
  python scripts/bc_train.py --episodes 100 --epochs 30 --out bc_policy.json
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.expert import GreedyZigzagExpert
from src.snake_game import SnakeEnv, STATE_DIM
from src.ppo import BottleneckPolicy


def parse_args():
    p = argparse.ArgumentParser(description="BC 预训练贪吃蛇策略")
    p.add_argument("--episodes", type=int, default=100, help="专家对局数")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=30, help="训练轮数")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=56)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="bc_policy.json")
    p.add_argument("--data", type=str, default=None,
                   help="已有数据文件，跳过收集")
    return p.parse_args()


def main():
    args = parse_args()
    # 固定全局 RNG（专家数据、BC minibatch shuffle 均可复现）
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    expert = GreedyZigzagExpert(seed=args.seed)
    env = SnakeEnv(seed=args.seed)

    if args.data and os.path.exists(args.data):
        d = np.load(args.data, allow_pickle=True)
        X, Y = d["states"], d["actions"]
        if X.ndim != 2 or X.shape[1] != STATE_DIM:
            raise ValueError(
                f"数据状态维度错误: 期望 (*, {STATE_DIM}), 实际 {X.shape}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"样本数不一致: states {X.shape[0]} vs actions {Y.shape[0]}")
        if X.shape[0] == 0:
            raise ValueError("数据为空")
        if Y.ndim != 1 or not set(np.unique(Y)).issubset({0, 1, 2}):
            raise ValueError(f"动作取值非法: {np.unique(Y)}")
        print(f"[bc] 载入数据 {X.shape[0]} 样本")
    else:
        print(f"[bc] 收集专家数据（{args.episodes} 局）...")
        X, Y, stats = expert.collect_bc_data(
            env, n_episodes=args.episodes, max_steps=args.max_steps)
        avg_food = np.mean([s[0] for s in stats])
        avg_len = np.mean([s[1] for s in stats])
        print(f"[bc] 收集完成: {X.shape[0]} 样本, "
              f"平均食物 {avg_food:.1f} 长度 {avg_len:.1f}")
        if args.data:
            np.savez(args.data, states=X, actions=Y)

    net = BottleneckPolicy(STATE_DIM, 3, hidden=args.hidden,
                           n_blocks=args.blocks, seed=args.seed)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    n = X.shape[0]
    idx = np.arange(n)
    for epoch in range(args.epochs):
        np.random.shuffle(idx)
        total_loss = 0.0
        total_acc = 0.0
        nb = 0
        for start in range(0, n, args.batch_size):
            mb = idx[start:start + args.batch_size]
            x = net._to_tensor(X[mb])
            y = torch.from_numpy(Y[mb]).to(net.device)
            logits, _ = net.forward(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_acc += (logits.argmax(-1) == y).float().mean().item()
            nb += 1
        print(f"[bc] epoch {epoch+1}/{args.epochs} loss="
              f"{total_loss/nb:.4f} acc={total_acc/nb:.3f}")

    net.save(args.out)
    print(f"[bc] 模型已保存: {args.out}")


if __name__ == "__main__":
    main()
