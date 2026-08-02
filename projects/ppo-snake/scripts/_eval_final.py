"""最终模型严格评估：20 局固定 seed，对比 masked / no-mask（step 滤波兜底）两种部署形态。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.ppo import MLPPolicy
from src.hamilton import HamiltonSnakeEnv

model = sys.argv[1] if len(sys.argv) > 1 else "experiments/final_bc/best_snake.json"
net = MLPPolicy(23, 3, hidden=128, seed=0)
net.load(model)

def evaluate(masked, n_ep=20):
    lens, foods, ints = [], [], []
    for i in range(n_ep):
        env = HamiltonSnakeEnv(seed=100 + i)
        s = env.reset(); done, steps = False, 0
        while not done and steps < 20000:
            a, _, _, _ = net.act(s, mask=env.safe_mask if masked else None)
            s, r, done, info = env.step(int(a))
            steps += 1
        lens.append(info["length"]); foods.append(info["food_eaten"])
        ints.append(env.n_intervened / max(env.n_steps, 1))
    L = np.array(lens); F = np.array(foods)
    print(f"{'masked  ' if masked else 'no-mask '} {n_ep} 局: "
          f"长度 平均{L.mean():5.1f} 中位{np.median(L):5.1f} min{int(L.min()):3d} max{int(L.max()):3d} | "
          f"食物 {F.mean():5.1f} | 干预率 {np.mean(ints)*100:4.1f}%")
    return L, F

evaluate(True)
evaluate(False)
