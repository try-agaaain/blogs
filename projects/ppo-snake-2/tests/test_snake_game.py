"""ppo-snake-2 单环境测试：sim_rel 状态编码、弧长插值、相对动作、奖励。"""

import unittest

import numpy as np

from src.snake_game import (SnakeEnv, STATE_DIM, WIDTH, HEIGHT, FRONT_SCAN,
                            BODY_SEG_N, STEP_LIMIT, REWARD_EAT, REWARD_HIT_WALL,
                            REWARD_HIT_BODY, _TURN_LEFT, _TURN_RIGHT,
                            DIR_INDEX, ABS_ACTION)


class TestState(unittest.TestCase):
    def test_state_dim(self):
        env = SnakeEnv(seed=0)
        s = env.reset()
        self.assertEqual(s.shape, (STATE_DIM,))
        self.assertEqual(STATE_DIM, FRONT_SCAN * 3 * 2 + BODY_SEG_N * 2 + 2)

    def test_state_range(self):
        """探测 ∈ [0,1]；蛇身段点/食物为旋转相对坐标，允许 [-1,1]。"""
        scan_dim = FRONT_SCAN * 3 * 2        # 探测维度起点
        env = SnakeEnv(seed=0)
        env.reset()
        for _ in range(200):
            a = np.random.randint(3)
            s, r, d, _ = env.step(a)
            # 探测 0~scan_dim-1 应在 [0,1]
            self.assertTrue(np.all(s[:scan_dim] >= 0) and
                            np.all(s[:scan_dim] <= 1),
                            f"探测越界: {s[:scan_dim]}")
            # 蛇身段点、食物方向为旋转相对坐标，在 [-1,1]
            self.assertTrue(np.all(s[scan_dim:] >= -1) and
                            np.all(s[scan_dim:] <= 1),
                            f"相对坐标越界: {s[scan_dim:]}")
            if d:
                env.reset()
        env = SnakeEnv(seed=0)
        s = env.reset()
        # 探测格障碍编码
        for i in range(0, scan_dim, 2):
            self.assertIn(s[i + 1], [0.0, 0.5, 1.0])

    def test_initial_state_shape(self):
        """初始：蛇头(4,6)朝右，三线探测布局正确（每线 FRONT_SCAN 格）。"""
        env = SnakeEnv(seed=0)
        s = env.reset()
        # 直行线（朝右）：k=1 是 (5,6)，初始空，block=0
        # 探测布局: 直行 FRONT_SCAN 格(has_food,block) + 左转 + 右转
        # 直行第1格 k=1: index 0,1
        self.assertEqual(s[0], 0.0)   # 无食物
        self.assertEqual(s[1], 0.0)   # 空
        # 直行第3格 k=3: index 4,5（(7,6) 空）
        self.assertEqual(s[4], 0.0)
        self.assertEqual(s[5], 0.0)


class TestBodySegments(unittest.TestCase):
    def test_arc_length_interpolation(self):
        """弧长插值：直线蛇时 9 段采样点沿身体均匀分布。"""
        env = SnakeEnv(seed=0)
        # 构造直线蛇：头(4,6) → (3,6) → (2,6) → (1,6)（4 节，n_body=3）
        env.snake = [(4, 6), (3, 6), (2, 6), (1, 6)]
        env.occ[:] = 0
        for (x, y) in env.snake:
            env.occ[y, x] = 1
        env.occ[6, 4] = 2
        env.direction = "right"
        s = env._get_state()
        seg0 = FRONT_SCAN * 3 * 2           # 探测维度起点
        seg = s[seg0:seg0 + BODY_SEG_N * 2].reshape(BODY_SEG_N, 2)
        # 前向轴=(1,0)，左向轴=(0,1)。身体点 (3,6),(2,6),(1,6)
        # 相对头 (-1,0),(-2,0),(-3,0)；y 相同故左向分量恒 0
        # j=0: t=0.5*3/9=1/6 → 插值距头 1.1667 格 → 前向=-1.1667/12
        self.assertAlmostEqual(seg[0, 0], -1.1667 / 12, places=3)
        self.assertAlmostEqual(seg[0, 1], 0.0, places=4)
        # j=8: t=8.5*3/9=2.833 ≥ n_body-1 → 取尾 (1,6)，距头 3 格
        self.assertAlmostEqual(seg[8, 0], -3.0 / 12, places=4)
        self.assertAlmostEqual(seg[8, 1], 0.0, places=4)

    def test_short_snake_padding(self):
        """蛇短于 9 段时，采样点仍连续（弧长插值不越界）。"""
        env = SnakeEnv(seed=0)
        env.reset()     # 初始 3 节（n_body=2）
        s = env._get_state()
        seg0 = FRONT_SCAN * 3 * 2           # 探测维度起点
        seg = s[seg0:seg0 + BODY_SEG_N * 2].reshape(BODY_SEG_N, 2)
        # 前向=(1,0)，身体相对头 (-1,0),(-2,0)
        # j=0: t=0.5*2/9=0.111 → 距头 1.111 格 → -1.111/12
        self.assertAlmostEqual(seg[0, 0], -1.1111 / 12, places=2)
        # j=8: t=8.5*2/9=1.889 ≥ n_body-1 → 取尾，距头 2 格
        self.assertAlmostEqual(seg[8, 0], -2.0 / 12, places=3)


class TestAction(unittest.TestCase):
    def test_rel_actions_turn(self):
        """相对动作：0=直行 1=左转 2=右转，方向正确变化。"""
        env = SnakeEnv(seed=0)
        env.reset()
        # 初始朝右 (3)
        self.assertEqual(DIR_INDEX[env.direction], 3)
        # 左转 → 朝上 (0)
        env.step(1)
        self.assertEqual(DIR_INDEX[env.direction], 0)
        # 直行 → 仍朝上
        env.step(0)
        self.assertEqual(DIR_INDEX[env.direction], 0)
        # 右转 → 朝右 (3)
        env.step(2)
        self.assertEqual(DIR_INDEX[env.direction], 3)

    def test_rel_vs_abs_turn_tables(self):
        """相对转向表正确：左转/右转后方向。"""
        # up(0)左转=left(2), up右转=right(3)
        self.assertEqual(_TURN_LEFT[0], 2)
        self.assertEqual(_TURN_RIGHT[0], 3)
        # down(1)左转=right(3), down右转=left(2)
        self.assertEqual(_TURN_LEFT[1], 3)
        self.assertEqual(_TURN_RIGHT[1], 2)
        # left(2)左转=down(1), left右转=up(0)
        self.assertEqual(_TURN_LEFT[2], 1)
        self.assertEqual(_TURN_RIGHT[2], 0)
        # right(3)左转=up(0), right右转=down(1)
        self.assertEqual(_TURN_LEFT[3], 0)
        self.assertEqual(_TURN_RIGHT[3], 1)


class TestReward(unittest.TestCase):
    def test_eat_reward(self):
        """吃食物 +2 且变长。"""
        env = SnakeEnv(seed=0)
        env.reset()
        # 把食物放到蛇头正前方（朝右，前方(5,6)）
        env.food = (5, 6)
        env.occ[6, 5] = 3
        s, r, d, info = env.step(0)    # 直行
        self.assertEqual(r, REWARD_EAT)
        self.assertEqual(len(env.snake), 4)
        self.assertEqual(info["food_eaten"], 1)

    def test_wall_death(self):
        """撞墙 -0.5。"""
        env = SnakeEnv(seed=0)
        env.reset()
        # 蛇在 (0,6) 朝左时撞墙
        env.snake = [(0, 6), (1, 6), (2, 6)]
        env.occ[:] = 0
        for (x, y) in env.snake:
            env.occ[y, x] = 1
        env.occ[6, 0] = 2
        env.direction = "left"
        env.food = None
        s, r, d, _ = env.step(0)       # 直行 → 撞左墙
        self.assertEqual(r, REWARD_HIT_WALL)
        self.assertTrue(d)

    def test_body_death(self):
        """撞自己身体 -0.5。"""
        env = SnakeEnv(seed=0)
        env.reset()
        # U 形蛇：头(3,6)朝右，前方(4,6)是身体
        env.snake = [(3, 6), (3, 5), (4, 5), (4, 6), (5, 6)]
        env.occ[:] = 0
        for (x, y) in env.snake:
            env.occ[y, x] = 1
        env.occ[6, 3] = 2
        env.direction = "right"
        env.food = None
        s, r, d, _ = env.step(0)       # 直行撞身体(4,6)
        self.assertEqual(r, REWARD_HIT_BODY)
        self.assertTrue(d)

    def test_tail_follow(self):
        """追尾：头能进即将移开的蛇尾格。"""
        env = SnakeEnv(seed=0)
        env.reset()
        # 蛇尾在 (2,6)，蛇头朝左，前方是尾 (2,6)（未吃食物时尾移开）
        env.snake = [(3, 6), (4, 6), (2, 6)]
        env.occ[:] = 0
        for (x, y) in env.snake:
            env.occ[y, x] = 1
        env.occ[6, 3] = 2
        env.direction = "left"
        env.food = None
        s, r, d, _ = env.step(0)       # 直行朝左 → (2,6) 是尾
        self.assertFalse(d)
        self.assertEqual(r, 0.0)
        # 新蛇头应在 (2,6)
        self.assertEqual(env.snake[0], (2, 6))

    def test_step_limit(self):
        """步数上限：到达 STEP_LIMIT 后正常结束（无惩罚）。"""
        env = SnakeEnv(seed=0)
        env.reset()
        env.food = None                 # 避免吃食物干扰
        env.steps = STEP_LIMIT - 1
        s, r, d, _ = env.step(0)        # 直行（前方为空）
        self.assertTrue(d)
        self.assertEqual(r, 0.0)
        self.assertEqual(len(env.snake), 3)


if __name__ == "__main__":
    unittest.main()
