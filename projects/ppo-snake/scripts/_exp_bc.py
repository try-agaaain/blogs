"""实验：纯 BC（行为克隆）学哈密顿专家，评估其吃食物/长蛇能力。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.ppo import MLPPolicy, AdamOptimizer, bc_update
from src.hamilton import HamiltonSnakeEnv

# 收集专家数据
def collect(n_episodes, eps_bc, seed, max_steps=1500):
    from src.hamilton import heuristic_action
    rng = np.random.RandomState(seed)
    states, actions, returns = [], [], []
    for i in range(n_episodes):
        env = HamiltonSnakeEnv(seed=seed * 1000 + i)
        s = env.reset()
        done, steps = False, 0
        es, ea, er = [], [], []
        while not done and steps < max_steps:
            a = heuristic_action(env.snake, env.direction, env.food,
                                 env.path, env.pos, path_dir=env.path_dir) \
                if rng.rand() >= eps_bc else rng.randint(3)
            es.append(s); ea.append(a)
            s, r, done, _ = env.step(a)
            er.append(r); steps += 1
        g = 0.0; er2 = np.zeros(len(er))
        for t in reversed(range(len(er))):
            g = er[t] + 0.99 * g; er2[t] = g
        states += es; actions += ea; returns += list(er2)
    return (np.array(states, np.float32), np.array(actions, np.int64),
            np.array(returns, np.float32))

def evaluate(net, n_ep=10):
    lens, foods, ints = [], [], []
    for i in range(n_ep):
        env = HamiltonSnakeEnv(seed=100 + i)
        s = env.reset(); done, steps = False, 0
        while not done and steps < 8000:
            a, _, _, _ = net.act(s)
            s, r, done, info = env.step(int(a))
            steps += 1
        lens.append(info["length"]); foods.append(info["food_eaten"])
        ints.append(env.n_intervened / max(env.n_steps, 1))
    return np.mean(lens), np.mean(foods), np.mean(ints)

for iters in [10, 30, 60, 120]:
    net = MLPPolicy(23, 3, hidden=128, seed=0)
    opt = AdamOptimizer(net.parameters(), lr=1e-3)
    states, actions, returns = collect(48, 0.05, 0)
    print(f"数据量: {len(states)} 条")
    for it in range(1, iters + 1):
        bc_update(net, opt, states, actions, returns=returns, label_smooth=0.1, val_coef=0.3)
    l, f, i = evaluate(net)
    print(f"BC {iters} 次: 平均身长 {l:.1f} | 食物 {f:.1f} | 干预率 {i*100:.1f}%")
