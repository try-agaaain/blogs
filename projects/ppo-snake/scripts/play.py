"""
演示脚本：加载训练好的 PPO 模型，让 AI 自动玩贪吃蛇

运行：
    python play.py                  # 终端动画演示一局
    python play.py --episodes 20    # 跑 20 局统计平均表现
    python play.py --model checkpoint/best_snake.json --interval 0.1
"""

import argparse
import os
import sys
import time

import numpy as np

# 保证从项目任意位置运行都能导入 src 包
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.snake_env import SnakeEnv, STATE_DIM
from src.ppo import MLPPolicy

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def play_one(net, interval=0.08, render=True, max_steps=3000):
    """让 AI 玩一局，返回 (总得分, 食物数, 长度, 步数)。"""
    env = SnakeEnv()
    s = env.reset()
    total_r, steps, done = 0.0, 0, False
    while not done:
        a, _, _, _ = net.act(s)
        s, r, done, info = env.step(a)
        total_r += r
        steps += 1
        if render and interval > 0:
            os.system("")  # 启用 Windows 终端 ANSI 转义
            sys.stdout.write("\033[H\033[2J")           # 清屏
            sys.stdout.write(env.render_text() + "\n")
            sys.stdout.write(f"得分 {total_r:+7.2f} | 食物 {info['food_eaten']:2d} | "
                             f"长度 {info['length']:2d} | 步数 {steps:4d}\n")
            sys.stdout.flush()
            time.sleep(interval)
        if steps >= max_steps:
            break
    return total_r, info["food_eaten"], info["length"], steps


def main():
    parser = argparse.ArgumentParser(description="加载 PPO 模型自动玩贪吃蛇")
    parser.add_argument("--model", type=str, default="checkpoint/best_snake.json")
    parser.add_argument("--episodes", type=int, default=1, help="演示局数")
    parser.add_argument("--interval", type=float, default=0.08, help="动画刷新间隔(秒)")
    parser.add_argument("--hidden", type=int, default=64, help="隐藏层宽度（须与训练一致）")
    parser.add_argument("--separate-critic", action="store_true",
                        help="模型使用独立价值头（训练时加了 --separate-critic 就需加此项）")
    args = parser.parse_args()

    assert os.path.exists(args.model), f"模型不存在: {args.model}，请先运行 python train.py"
    net = MLPPolicy(STATE_DIM, 3, hidden=args.hidden, seed=0,
                    separate_critic=args.separate_critic)
    net.load(args.model)

    print(f"加载模型: {args.model}  (隐藏层={args.hidden})")
    print("=" * 50)

    total_r = total_food = total_len = 0
    for ep in range(args.episodes):
        r, food, length, steps = play_one(net, interval=args.interval,
                                          render=(args.episodes == 1))
        total_r += r; total_food += food; total_len += length
        if args.episodes > 1:
            print(f"第 {ep+1:2d} 局: 得分 {r:+8.2f} | 食物 {food:2d} | 长度 {length:2d}")

    if args.episodes > 1:
        print("=" * 50)
        print(f"平均得分 {total_r/args.episodes:+8.2f} | 平均食物 {total_food/args.episodes:.1f} | "
              f"平均长度 {total_len/args.episodes:.1f}")


if __name__ == "__main__":
    main()
