"""ppo-snake-2 向量环境测试：单/向量一致性、奖励、长蛇边界。"""

import unittest

import numpy as np

from src.snake_game import (SnakeEnv, STATE_DIM, STEP_LIMIT, REWARD_EAT,
                            FRONT_SCAN, BODY_SEG_N, _TURN_LEFT, _TURN_RIGHT)
from src.snake_vec import VectorSnakeEnv, _MAXLEN


class TestVectorConsistency(unittest.TestCase):
    """向量环境与单环境在相同动作序列下必须逐格一致。"""

    def test_initial_state_consistent(self):
        for seed in range(5):
            env = SnakeEnv(seed=seed)
            s = env.reset()
            vec = VectorSnakeEnv(n_envs=1, seeds=[seed])
            sv = vec.get_states()[0]
            self.assertTrue(np.allclose(s, sv), f"seed{seed} 初始状态不一致")

    def test_full_trajectory_consistent(self):
        """同 seed 随机动作序列，活步状态/reward/done 完全一致。"""
        np.random.seed(0)
        for trial in range(4):
            seed = trial * 50 + 3
            env = SnakeEnv(seed=seed)
            vec = VectorSnakeEnv(n_envs=1, seeds=[seed])
            s = env.reset()
            sv = vec.get_states()[0]
            self.assertTrue(np.allclose(s, sv))
            for step in range(2000):
                a = np.random.randint(3)
                s, r, d, _ = env.step(a)
                sv, rv, dv, _ = vec.step(np.array([a]))
                self.assertEqual(d, dv, f"trial{trial} step{step} done 不一致")
                if d:
                    # 死亡步单环境返回占位状态，跳过比较并同步 reset
                    s = env.reset()
                    sv = vec.reset(np.array([0]))[0]
                    self.assertTrue(np.allclose(s, sv),
                                    f"trial{trial} step{step} reset 后不一致")
                    continue
                self.assertTrue(np.allclose(s, sv, atol=1e-6),
                                f"trial{trial} step{step} 状态不一致")
                self.assertEqual(r, rv,
                                 f"trial{trial} step{step} reward 不一致")

    def test_long_snake_arc_consistency(self):
        """构造超长蛇形，验证弧长插值在长蛇下单/向量一致。"""
        from src.snake_game import SnakeEnv
        env = SnakeEnv(seed=7)
        vec = VectorSnakeEnv(n_envs=1, seeds=[7])
        s = env.reset()
        sv = vec.get_states()[0]
        self.assertTrue(np.allclose(s, sv))

        # 构造 40 节蛇形（行蛇形弯折），同一布局写到单/向量
        pts = []
        for row in range(8):
            if row % 2 == 0:
                pts += [(x, row) for x in range(12)]
            else:
                pts += [(11 - x, row) for x in range(12)]
        pts = pts[:40]
        hx, hy = pts[0]
        # 单环境
        env.snake = list(pts)
        env.occ[:] = 0
        for (x, y) in pts:
            env.occ[y, x] = 1
        env.occ[hy, hx] = 2
        env.direction = "right"
        env.food = (5, 10)
        env.occ[10, 5] = 3
        # 向量环境
        i = 0
        vec.heads[i] = pts[0]
        for j, (x, y) in enumerate(pts):
            vec.body[i, j] = (x, y)
        vec.lens[i] = len(pts)
        vec.dirs[i] = 3
        vec.foods[i] = (5, 10)
        vec.occ[i].fill(0)
        for (x, y) in pts:
            vec.occ[i, y, x] = 1
        vec.occ[i, 10, 5] = 3
        vec.occ[i, hy, hx] = 2

        s = env._get_state()
        sv = vec.get_states()[0]
        self.assertTrue(np.allclose(s, sv, atol=1e-5),
                        "长蛇(40节)弧长插值 单/向量不一致")

        # 再走一步验证步进后仍一致
        s, r, d, _ = env.step(0)
        sv, rv, dv, _ = vec.step(np.array([0]))
        self.assertTrue(np.allclose(s, sv, atol=1e-5),
                        "长蛇步进后状态不一致")
        self.assertEqual(r, rv)


class TestVectorBatch(unittest.TestCase):
    def test_batch_shape(self):
        vec = VectorSnakeEnv(n_envs=8, seeds=list(range(8)))
        s = vec.get_states()
        self.assertEqual(s.shape, (8, STATE_DIM))

    def test_batch_independent(self):
        """多环境并行各自独立演化，状态互不干扰。"""
        n = 8
        vec = VectorSnakeEnv(n_envs=n, seeds=list(range(n)))
        states = vec.get_states()
        self.assertEqual(states.shape, (n, STATE_DIM))
        # 每个环境的蛇头都应在初始位置
        for i in range(n):
            self.assertEqual(tuple(vec.heads[i]), (4, 6))

    def test_batch_reset_partial(self):
        """只重置部分环境，其余保持。"""
        vec = VectorSnakeEnv(n_envs=4, seeds=list(range(4)))
        s0 = vec.get_states()
        heads0 = vec.heads.copy()
        # 让所有环境走几步
        for _ in range(10):
            vec.step(np.random.randint(3, size=4))
        moved = ~np.all(vec.heads == heads0, axis=1)
        self.assertTrue(moved.any(), "10步后至少一个环境应移动")
        # 重置 0,2 号
        vec.reset(np.array([0, 2]))
        s2 = vec.get_states()
        # reset 后蛇布局与初始一致，但食物位置重新随机（探测/食物方向维度不同）。
        # 蛇身段点维（探测之后）与初始一致即可
        seg0 = FRONT_SCAN * 3 * 2
        seg1 = seg0 + BODY_SEG_N * 2
        self.assertTrue(np.allclose(s2[0][seg0:seg1], s0[0][seg0:seg1]))
        self.assertTrue(np.allclose(s2[2][seg0:seg1], s0[2][seg0:seg1]))
        self.assertTrue(np.array_equal(vec.heads[0], heads0[0]))
        self.assertTrue(np.array_equal(vec.heads[2], heads0[2]))


class TestLongSnakeBoundary(unittest.TestCase):
    """修复旧项目 bug：长蛇接近满盘时吃食物不越界。"""

    def test_maxlen_buffer_safe(self):
        """蛇身缓冲上限 _MAXLEN 足够容纳满盘 + 4 格余量。"""
        self.assertGreaterEqual(_MAXLEN, 12 * 12)
        self.assertEqual(_MAXLEN, 148)

    def test_full_board_eat(self):
        """蛇长 140 时吃食物增长：缓冲安全、奖励与 info 正确。"""
        vec = VectorSnakeEnv(n_envs=1, seeds=[0])
        env_i = 0
        # 构造 140 格蛇形：列 0 竖直 8 格 + 其余 11 列逐行蛇形，
        # 头 (0,4) 朝上，正前方 (0,3) 留空放食物
        pts = [(0, y) for y in range(4, 12)]
        for k in range(11, -1, -1):
            if k % 2 == 1:
                pts += [(x, k) for x in range(1, 12)]
            else:
                pts += [(x, k) for x in range(11, 0, -1)]
        assert len(pts) == 140
        vec.lens[env_i] = 140
        vec.food_eaten[env_i] = 137
        for j, (x, y) in enumerate(pts):
            vec.body[env_i, j] = (x, y)
        vec.heads[env_i] = pts[0]
        vec.dirs[env_i] = 0                    # up
        vec.foods[env_i] = (0, 3)
        vec.occ[env_i].fill(0)
        for (x, y) in pts:
            vec.occ[env_i, y, x] = 1
        vec.occ[env_i, 3, 0] = 3               # 食物
        vec.occ[env_i, 4, 0] = 2               # 头

        s, r, d, info = vec.step(np.array([0]))   # 直行向上吃食物
        self.assertEqual(r[0], REWARD_EAT)
        self.assertEqual(vec.lens[env_i], 141)    # 增长且缓冲不越界
        self.assertFalse(d[0])
        self.assertEqual(info[0]["food_eaten"], 138)
        self.assertEqual(info[0]["length"], 141)
        self.assertEqual(s.shape, (1, STATE_DIM))
        # 新头应落在食物格
        self.assertTrue(np.array_equal(vec.heads[env_i], (0, 3)))

    def test_step_limit_vector(self):
        """向量环境步数上限：到达 STEP_LIMIT 结束且无惩罚。"""
        vec = VectorSnakeEnv(n_envs=1, seeds=[0])
        vec.steps[0] = STEP_LIMIT - 1
        s, r, d, _ = vec.step(np.array([0]))   # 直行，前方为空
        self.assertTrue(d[0])
        self.assertIn(r[0], (0.0, REWARD_EAT))
        self.assertEqual(vec.steps[0], STEP_LIMIT)


class TestDeathStepConsistency(unittest.TestCase):
    """死亡步后死环境状态必须与单环境逐元素一致。"""

    def test_death_step_state_consistent(self):
        np.random.seed(3)
        env = SnakeEnv(seed=3)
        vec = VectorSnakeEnv(n_envs=1, seeds=[3])
        env.reset()
        vec.get_states()
        saw_death = False
        for _ in range(1200):
            a = np.random.randint(3)
            s, r, d, _ = env.step(a)
            sv, rv, dv, _ = vec.step(np.array([a]))
            self.assertEqual(d, dv)
            if d:
                saw_death = True
                self.assertTrue(np.allclose(s, sv, atol=1e-6),
                                "死亡步后死环境状态不一致")
                self.assertEqual(r, rv)
                env.reset()
                vec.reset(np.array([0]))
        self.assertTrue(saw_death, "测试应至少出现一次死亡")


if __name__ == "__main__":
    unittest.main()
