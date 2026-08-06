"""ppo-snake-2 贪心 Z 字专家：BC 预训练数据来源。

决策逻辑与旧项目一致：
  1. 枚举 3 个相对动作对应绝对方向，过滤安全方向（不撞墙/身，追尾允许）
  2. 当前方向安全则优先保持（Z 字折叠）
  3. 否则选最靠近食物的安全方向
  4. 全危险时退回蛇头后方兜底
"""

import numpy as np

from src.snake_game import (DIR_INDEX, ABS_ACTION, _TURN_LEFT, _TURN_RIGHT,
                            abs_to_rel, _OCC_BODY, _OCC_HEAD)


class GreedyZigzagExpert:
    """贪心 Z 字专家。act(env) 返回绝对动作 0~3。"""

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def _safe(self, env, nx, ny, ate):
        if nx < 0 or nx >= env.width or ny < 0 or ny >= env.height:
            return False
        cell = int(env.occ[ny, nx])
        if cell in (_OCC_BODY, _OCC_HEAD):      # 身体 / 蛇头
            if ate:
                return False
            tx, ty = env.snake[-1]
            return (nx == tx and ny == ty)      # 追尾允许
        return True

    def _rank_actions(self, env):
        """返回按偏好排序的安全绝对动作列表。"""
        hx, hy = env.snake[0]
        food = env.food
        cur = DIR_INDEX[env.direction]

        def target_ate(nx, ny):
            return food is not None and (nx, ny) == food

        cands = []
        for a in range(4):
            dx, dy = ABS_ACTION[a][1]
            nx, ny = hx + dx, hy + dy
            if self._safe(env, nx, ny, target_ate(nx, ny)):
                cands.append(a)
        if not cands:
            return []

        cur_dist = (abs(hx - food[0]) + abs(hy - food[1])
                    if food is not None else 1 << 30)

        scored = []
        for a in cands:
            dx, dy = ABS_ACTION[a][1]
            nx, ny = hx + dx, hy + dy
            score = 0.0
            if food is not None:
                nxt_dist = abs(nx - food[0]) + abs(ny - food[1])
                score += (nxt_dist - cur_dist) * 10.0
            if a == cur:
                score -= 1.0
            scored.append((score, a))
        # 等分（score 相同）时随机打破，避免方向索引带来的系统性偏差
        scored.sort(key=lambda t: (t[0], self.rng.rand()))
        return [a for _, a in scored]

    def act(self, env):
        """返回绝对动作 0~3。"""
        ranked = self._rank_actions(env)
        if not ranked:
            cur = DIR_INDEX[env.direction]
            back = cur ^ 1               # 真反向（up↔down, left↔right）
            hx, hy = env.snake[0]
            dx, dy = ABS_ACTION[back][1]
            if self._safe(env, hx + dx, hy + dy, False):
                return back
            return cur
        return ranked[0]

    # ---------------- BC 数据收集 ----------------
    def collect_bc_data(self, env, n_episodes=50, max_steps=3000,
                        progress=True):
        """跑专家对局收集 (states, rel_actions)。掉头样本丢弃。"""
        all_states, all_actions, all_stats = [], [], []
        for ep in range(n_episodes):
            s = env.reset()
            for _ in range(max_steps):
                a = self.act(env)
                d = DIR_INDEX[env.direction]
                ra = int(abs_to_rel(np.array([a]), np.array([d]))[0])
                if ra < 0:
                    # 专家选了掉头：环境会忽略（相对动作不存在），
                    # 转为环境忽略掉的合法相对动作直行
                    s, r, done, info = env.step(0)
                    if done:
                        break
                    continue
                all_states.append(s)
                all_actions.append(ra)
                s, r, done, info = env.step(ra)
                if done:
                    break
            food_count = info.get("food_eaten", 0)
            all_stats.append((food_count, info.get("length", len(env.snake))))
            if progress and (ep + 1) % 10 == 0:
                avg_food = np.mean([st[0] for st in all_stats])
                avg_len = np.mean([st[1] for st in all_stats])
                print(f"[expert] 已收集 {len(all_states)} 样本 / {ep+1} 局, "
                      f"平均食物 {avg_food:.1f} 长度 {avg_len:.1f}")
        return (np.asarray(all_states, dtype=np.float32),
                np.asarray(all_actions, dtype=np.int64),
                all_stats)
