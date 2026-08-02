"""
行为克隆预训练：用启发式专家打局收集 (状态, 动作) 数据，
监督训练策略网络，得到"天生就会玩"的初始权重，再交给 PPO 微调。

为什么能降低种子敏感性？
  随机初始化从均匀分布开始，策略能不能学会取决于初始方向是否偶然碰到
  食物（0.03% 正样本）——这就是"种子敏感性"的根源。
  预训练后策略一开始就会"追食物 + 避障碍"，PPO 只需微调，
  不再依赖随机初始化撞大运。

用法：
    python pretrain.py --iters 40 --save checkpoint/pretrained.json
    python pretrain.py --eps-bc 0.1    # 数据收集时 10% 概率随机动作（增加多样性）
"""
import argparse
import os
import sys
import time

import numpy as np

# 保证从项目任意位置运行都能导入 src 包
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.snake_game import SnakeEnv, STATE_DIM, TURN, DIR_VEC
from src.ppo import MLPPolicy, AdamOptimizer, bc_update


def safety_steps(env, d):
    """沿方向 d 还能安全走几格（撞墙/撞身体前）。"""
    dx, dy = DIR_VEC[d]
    x, y = env.snake[0]
    body = set(env.snake)
    steps = 0
    while True:
        x, y = x + dx, y + dy
        if x < 0 or x >= env.width or y < 0 or y >= env.height or (x, y) in body:
            return steps
        steps += 1


def heuristic_action(env):
    """贪心专家：优先"离食物近 + 前方安全通道长"，全撞时选能活最久的。"""
    hx, hy = env.snake[0]
    body = set(env.snake)
    fx, fy = env.food
    cands = []
    for a in (0, 1, 2):
        nd = TURN[env.direction][a]
        dx, dy = DIR_VEC[nd]
        nx, ny = hx + dx, hy + dy
        if nx < 0 or nx >= env.width or ny < 0 or ny >= env.height or (nx, ny) in body:
            continue
        dist = abs(nx - fx) + abs(ny - fy)
        cands.append((a, dist, -safety_steps(env, nd)))
    if cands:
        return min(cands, key=lambda c: (c[1], c[2]))[0]
    return max(range(3), key=lambda a: safety_steps(env, TURN[env.direction][a]))


def collect_data(n_episodes=400, eps_bc=0.1, seed=0):
    """用启发式打 n_episodes 局，返回 (states, actions, returns)。

    returns 是折扣回报（γ=0.99）：启发式每局能活到身长 25，这给出了
    每个状态"好到什么程度"的标签，用来训练 Critic。
    """
    rng = np.random.RandomState(seed)
    states, actions, returns = [], [], []
    for i in range(n_episodes):
        env = SnakeEnv(seed=seed * 1000 + i)
        s = env.reset()
        done = False
        ep_states, ep_actions, ep_rewards = [], [], []
        while not done:
            if rng.rand() < eps_bc:
                a = rng.randint(3)
            else:
                a = heuristic_action(env)
            ep_states.append(s)
            ep_actions.append(a)
            s, r, done, _ = env.step(a)
            ep_rewards.append(r)
        # 从局尾反向计算折扣回报
        ep_returns = np.zeros(len(ep_rewards))
        g = 0.0
        for t in reversed(range(len(ep_rewards))):
            g = ep_rewards[t] + 0.99 * g
            ep_returns[t] = g
        states.extend(ep_states)
        actions.extend(ep_actions)
        returns.extend(ep_returns)
    return (np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(returns, dtype=np.float32))


def evaluate(net, n_episodes=24):
    """用网络自身玩 n_episodes 局，返回平均食物数与平均长度。"""
    total_food = total_len = 0
    for i in range(n_episodes):
        env = SnakeEnv()
        s = env.reset()
        done = False
        while not done:
            a, _, _, _ = net.act(s)
            s, _, done, info = env.step(a)
        total_food += info["food_eaten"]
        total_len += info["length"]
    return total_food / n_episodes, total_len / n_episodes


def bc_update_wrapper(net, opt, states, actions, returns, label_smooth=0.1):
    """给 train.py 用的 BC 更新包装：价值头随策略一起训练（独立 Critic 时互不干扰）。"""
    from src.ppo import bc_update
    return bc_update(net, opt, states, actions, returns=returns,
                     label_smooth=label_smooth, val_coef=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=40, help="预训练轮数（每轮一批数据）")
    parser.add_argument("--episodes", type=int, default=400, help="每轮收集多少局")
    parser.add_argument("--eps-bc", type=float, default=0.1, help="数据收集随机扰动概率")
    parser.add_argument("--lr", type=float, default=1e-2, help="预训练学习率")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="checkpoint/pretrained.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    net = MLPPolicy(STATE_DIM, 3, hidden=args.hidden, seed=args.seed,
                    separate_critic=True)
    opt = AdamOptimizer(net.parameters(), lr=args.lr)

    t0 = time.time()
    for it in range(1, args.iters + 1):
        states, actions, returns = collect_data(args.episodes, args.eps_bc, args.seed)
        # 独立 Critic：策略用纯 BC 学，价值头同时用折扣回报监督学
        loss = bc_update(net, opt, states, actions, returns=returns,
                         label_smooth=0.1, val_coef=0.3)
        if it % 5 == 0 or it == 1:
            food, length = evaluate(net)
            print(f"[bc|iter {it:3d}/{args.iters}] loss {loss:.4f} | "
                  f"评估: 食物 {food:5.2f} | 身长 {length:6.2f} | {time.time()-t0:6.1f}s")

    net.save(args.save)
    food, length = evaluate(net)
    print(f"\n预训练完成，模型已存 {args.save}")
    print(f"预训练后策略：平均食物 {food:.2f}，平均身长 {length:.1f}（启发式上界 22.6）")


if __name__ == "__main__":
    main()
