"""ppo-snake-2 贪心 Z 字专家测试：安全过滤、追尾允许、掉头样本丢弃。"""

import unittest

import numpy as np

from src.expert import GreedyZigzagExpert
from src.snake_game import (SnakeEnv, STATE_DIM, DIR_INDEX, ABS_ACTION,
                            abs_to_rel)


class TestExpertAction(unittest.TestCase):
    def _boxed(self, snake, direction):
        """构造指定蛇形并返回 env。"""
        env = SnakeEnv(seed=0)
        env.snake = list(snake)
        env.occ[:] = 0
        for (x, y) in env.snake:
            env.occ[y, x] = 1
        env.occ[snake[0][1], snake[0][0]] = 2
        env.direction = direction
        env.food = None
        return env

    def test_no_suicidal_action(self):
        """存在安全方向时，专家不得选立即撞墙/撞身的动作。"""
        for seed in range(5):
            env = SnakeEnv(seed=seed)
            env.reset()
            ex = GreedyZigzagExpert(seed=seed)
            for _ in range(300):
                a = ex.act(env)
                hx, hy = env.snake[0]
                dx, dy = ABS_ACTION[a][1]
                nx, ny = hx + dx, hy + dy
                # 本步是否存在至少一个安全方向（死局除外，此时任何选择都只能牺牲）
                safe_exists = any(
                    ex._safe(env, hx + ABS_ACTION[aa][1][0],
                             hy + ABS_ACTION[aa][1][1],
                             env.food is not None and
                             (hx + ABS_ACTION[aa][1][0],
                              hy + ABS_ACTION[aa][1][1]) == env.food)
                    for aa in range(4))
                if safe_exists:
                    if nx < 0 or nx >= env.width or ny < 0 or ny >= env.height:
                        self.fail(f"seed{seed} 有安全方向却选了撞墙动作 {a}")
                    if (nx, ny) in env.snake[:-1]:
                        self.fail(f"seed{seed} 有安全方向却选了撞身动作 {a}")
                # 转相对动作推进（专家兜底掉头时跳过，此情形极少）
                d = DIR_INDEX[env.direction]
                ra = int(abs_to_rel(np.array([a]), np.array([d]))[0])
                if ra < 0:
                    continue
                s, r, d, _ = env.step(ra)
                if d:
                    break

    def test_tail_follow_allowed(self):
        """追尾允许：头前方是即将移开的蛇尾格时应是安全方向。"""
        env = self._boxed([(3, 6), (4, 6), (2, 6)], "left")
        ex = GreedyZigzagExpert(seed=0)
        ranked = ex._rank_actions(env)
        self.assertIn(DIR_INDEX["left"], ranked)

    def test_collect_bc_data_shapes(self):
        """BC 数据形状与取值范围正确。"""
        env = SnakeEnv(seed=0)
        ex = GreedyZigzagExpert(seed=0)
        X, Y, stats = ex.collect_bc_data(env, n_episodes=5, max_steps=2000,
                                         progress=False)
        self.assertEqual(X.shape[1], STATE_DIM)
        self.assertEqual(X.shape[0], Y.shape[0])
        self.assertTrue(set(np.unique(Y)).issubset({0, 1, 2}))
        self.assertEqual(len(stats), 5)

    def test_uturn_samples_dropped(self):
        """专家掉头步不收集样本（相对动作空间不存在掉头），环境用直行推进。"""
        env = SnakeEnv(seed=0)
        env.reset()
        ex = GreedyZigzagExpert(seed=0)

        def fake_act(e):
            d = DIR_INDEX[e.direction]
            return d ^ 1                 # 总是返回真反向（掉头）

        ex.act = fake_act
        X, Y, _ = ex.collect_bc_data(env, n_episodes=1, max_steps=10,
                                     progress=False)
        # 掉头样本全部丢弃 → 没有可学习样本
        self.assertEqual(X.shape[0], 0)
        self.assertEqual(Y.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
