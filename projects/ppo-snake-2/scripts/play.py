"""ppo-snake-2 演示：用训练好的策略跑贪吃蛇（终端可视化）。"""

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.snake_game import SnakeEnv, STATE_DIM
from src.ppo import BottleneckPolicy


def clear_screen():
    """兼容 Windows 旧控制台与 ANSI 终端。"""
    if os.name == "nt":
        os.system("cls")
    else:
        print("\033[H\033[J", end="")


def render(env, step, total_score):
    clear_screen()
    print(f"step={step} length={len(env.snake)} food={env.food_eaten} "
          f"score={total_score}")
    for y in range(env.height):
        row = ""
        for x in range(env.width):
            c = int(env.occ[y, x])
            if c == 2:
                row += "H"
            elif c == 1:
                row += "#"
            elif c == 3:
                row += "F"
            else:
                row += "."
        print(row)
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="best_snake.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--hidden", type=int, default=56)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--sample", action="store_true",
                   help="按策略随机采样（默认贪心决策）")
    args = p.parse_args()

    # 优先从权重文件读取网络结构（hidden/n_blocks 与训练时一致）
    with open(args.weights, "r", encoding="utf-8") as f:
        meta = json.load(f)
    hidden = meta.get("hidden", args.hidden)
    blocks = meta.get("n_blocks", args.blocks)

    net = BottleneckPolicy(STATE_DIM, 3, hidden=hidden, n_blocks=blocks,
                           seed=0)
    net.load(args.weights)
    env = SnakeEnv(seed=args.seed)
    s = env.reset()
    total = 0.0
    step = 0
    try:
        while True:
            a, _, _, _ = net.act(s, greedy=not args.sample)
            s, r, done, info = env.step(a)
            total += r
            render(env, step, total)
            step += 1
            if done:
                print(f"结束: length={info['length']} "
                      f"food={info['food_eaten']}")
                break
            time.sleep(1.0 / args.fps)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
