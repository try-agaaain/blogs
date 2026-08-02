"""
录制回放脚本：让训练好的 PPO 模型完整打一局，把每一步的真实决策记录成动作序列，
供 HTML 交互演示（「AI 自动玩」）使用。

    python record_replay.py

输出 ai_replay.js，结构如下（回放时按环境规则逐步推演即可还原整局）：
    const AI_REPLAY = {
        snake:    [[5,6],[4,6],[3,6]],   // 初始蛇（头在前）
        dir:      "right",               // 初始方向
        food:     [6,2],                 // 初始食物位置
        actions:  [0,2,1,...],           // 每一步决策：0 直行 / 1 左转 / 2 右转
        newFoods: [[3,9],[4,0],...],     // 每次吃到食物后新放置的食物（按顺序）
    };

原理：贪吃蛇移动是确定性的——给定初始棋盘和动作序列，就能唯一还原整个局面。
所以只需要存动作序列（几百个整数，约 1KB），而不是逐帧的快照。
"""

import argparse
import json
import os
import sys

import numpy as np

from snake_env import SnakeEnv, STATE_DIM
from ppo import MLPPolicy

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def play_and_record(net):
    """让 AI 完整打一局，返回可还原整局的数据字典。"""
    env = SnakeEnv()
    s = env.reset()

    record = {
        "snake": [list(p) for p in env.snake],
        "dir": env.direction,
        "food": list(env.food),
        "actions": [],
        "newFoods": [],
    }

    while True:
        a, _, _, _ = net.act(s)
        record["actions"].append(int(a))
        s, r, done, info = env.step(a)
        if r > 5 and not done:          # 吃到食物后环境会重新放一个
            record["newFoods"].append(list(env.food))
        if done:
            break
        if len(record["actions"]) >= 5000:   # 保险丝，理论上到不了
            break

    record["_meta"] = {
        "steps": len(record["actions"]),
        "foods": env.food_eaten,
        "length": len(env.snake),
    }
    return record


def main():
    parser = argparse.ArgumentParser(description="录制 AI 完整一局的回放数据")
    parser.add_argument("--model", type=str, default="checkpoint/best_snake.npz")
    parser.add_argument("--hidden", type=int, default=64, help="隐藏层宽度（须与训练一致）")
    parser.add_argument("--trials", type=int, default=10,
                        help="跑多少局，挑玩得最长的一局作为回放")
    parser.add_argument("--separate-critic", action="store_true",
                        help="模型使用独立价值头（训练时加了 --separate-critic 就需加此项）")
    args = parser.parse_args()

    assert os.path.exists(args.model), f"模型不存在: {args.model}，请先运行 python train.py"
    net = MLPPolicy(STATE_DIM, 3, hidden=args.hidden, seed=0,
                    separate_critic=args.separate_critic)
    net.load(args.model)

    best = None
    for ep in range(args.trials):
        rec = play_and_record(net)
        meta = rec.pop("_meta")
        print(f"第 {ep+1:2d} 局: 步数 {meta['steps']:4d} | 食物 {meta['foods']:2d} | "
              f"身长 {meta['length']:2d}")
        if best is None or meta["steps"] > best[0]:
            best = (meta["steps"], rec)

    _, rec = best
    payload = {k: rec[k] for k in ("snake", "dir", "food", "actions", "newFoods")}
    js = "const AI_REPLAY = " + json.dumps(payload, separators=(",", ":")) + ";\n"

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_replay.js")
    with open(out, "w", encoding="utf-8") as f:
        f.write(js)
    print(f"\n已写入 {out}  ({len(js)//1024} KB, {len(payload['actions'])} 步)")


if __name__ == "__main__":
    main()
