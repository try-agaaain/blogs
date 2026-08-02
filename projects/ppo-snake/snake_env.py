"""
贪吃蛇环境 —— PPO 训练用

游戏规则（简洁版）：
  * 12x12 网格，蛇由若干"格子"组成，头部决定前进方向。
  * 每一时刻蛇只能做 3 种动作之一：0=直行, 1=左转, 2=右转（相对当前方向）。
  * 吃到食物 +10 分，蛇变长；撞墙 / 撞到自己 -10 分并结束本局。
  * 每走一步有 -0.05 的小惩罚（鼓励尽快吃到食物，防止绕圈刷步数）。

状态特征（21 维，长蛇生存导向）：
  0~2   三向是否立即危险（撞墙/身体）
  3~6   食物在哪个相对方向
  7~10  当前方向 one-hot
  11~13 三向到墙距离
  14    食物曼哈顿距离
  15    蛇长比例
  16~18 三向"安全通道长度"（沿该方向还能安全走几格）
  19    自由空间比例（剩余空格 / 总格数）
  20    头部被包围程度（蛇头周围一圈被占的比例）

奖励（稀疏 + 双 shaping）：
  * 吃食物 +20，撞墙 -5，撞自己 -8（区分对待），每步 -0.02
  * potential-based shaping A：靠近食物每格 +1.2
  * potential-based shaping B：BFS 可达空间变大（小奖励）——
    教蛇"别钻死胡同"，这是长蛇存活的关键
"""

import random

import numpy as np

WIDTH, HEIGHT = 12, 12          # 网格尺寸
N_DIR = 4                       # 绝对方向数（上/下/左/右）
STATE_DIM = 21                  # 状态特征维度
MAX_CELLS = WIDTH * HEIGHT      # 总格数

# 奖励系数
REWARD_EAT = 20.0
REWARD_HIT_WALL = -5.0
REWARD_HIT_BODY = -5.0          # 与原版一致，避免过重惩罚破坏探索
REWARD_STEP = -0.02
SHAPING_FOOD_COEF = 1.5         # 靠近食物 shaping（与原版一致）
SHAPING_REACH_COEF = 0.02       # 可达空间 shaping
SHAPING_REACH_MIN_LEN = 6       # 蛇长达到此值后才启用可达空间 shaping

# 绝对方向 -> 坐标增量
DIR_VEC = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
# 相对转向表：按 (当前方向, 动作) 给出新绝对方向
TURN = {
    "up":    {0: "up", 1: "left", 2: "right"},
    "down":  {0: "down", 1: "right", 2: "left"},
    "left":  {0: "left", 1: "down", 2: "up"},
    "right": {0: "right", 1: "up", 2: "down"},
}
# 绝对方向 -> one-hot 索引
DIR_INDEX = {"up": 0, "down": 1, "left": 2, "right": 3}


class SnakeEnv:
    """贪吃蛇环境，实现最小化 gym 风格接口：reset / step / render_text。"""

    def __init__(self, width=WIDTH, height=HEIGHT, seed=None):
        self.width = width
        self.height = height
        self.max_cells = width * height
        self.rng = random.Random(seed)
        self.snake = []          # [(x, y), ...] 头部在前
        self.direction = "right"
        self.food = None
        self.steps = 0
        self.food_eaten = 0
        self.done = False
        self.last_reward = 0.0
        self.info = {}
        self._last_reach = 0

    # ---------------------------------------------------------- 基础
    def _reset(self):
        # 蛇初始在中间偏左，长度 3，方向向右
        cx, cy = self.width // 2 - 2, self.height // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction = "right"
        self.steps = 0
        self.food_eaten = 0
        self.done = False
        self.last_reward = 0.0
        self._place_food()
        self.info = {"length": len(self.snake), "food_eaten": 0}
        # 记录当前可达空间（shaping B 的基准）
        self._last_reach = self._reachable_free()

    def _place_food(self):
        free = [(x, y) for x in range(self.width) for y in range(self.height)
                if (x, y) not in self.snake]
        self.food = self.rng.choice(free) if free else None

    def _step_limit(self):
        # 随蛇变长放宽步数上限，但总有限制（防止无限绕圈）
        return 200 + 50 * len(self.snake)

    # ---------------------------------------------------------- 核心状态
    def _get_state(self):
        """返回 21 维特征向量。"""
        hx, hy = self.snake[0]
        body = set(self.snake)

        def danger(d):
            dx, dy = DIR_VEC[d]
            nx, ny = hx + dx, hy + dy
            return 1.0 if (nx < 0 or nx >= self.width or ny < 0 or ny >= self.height
                           or (nx, ny) in body) else 0.0

        # 安全通道长度：沿方向一步步走，直到撞墙/身体，返回能走的格数（归一化）
        def safety_len(d):
            dx, dy = DIR_VEC[d]
            x, y = hx, hy
            steps = 0
            while True:
                x, y = x + dx, y + dy
                if x < 0 or x >= self.width or y < 0 or y >= self.height \
                        or (x, y) in body:
                    break
                steps += 1
            return steps / max(self.width, self.height)

        # 相对方向：直行 = 当前方向，左转 / 右转后的绝对方向
        fwd = self.direction
        left = TURN[self.direction][1]
        right = TURN[self.direction][2]

        fx, fy = self.food
        food_rel = [0.0, 0.0, 0.0, 0.0]   # [straight, left, right, back]

        def is_ahead(d):
            """判断食物是否在绝对方向 d 上（沿该方向走一定会撞到）。"""
            dx, dy = DIR_VEC[d]
            nx, ny = hx, hy
            while True:
                nx, ny = nx + dx, ny + dy
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    return False
                if (nx, ny) == (fx, fy):
                    return True
                if (nx, ny) in body:
                    return False

        food_rel[0] = 1.0 if is_ahead(fwd) else 0.0
        food_rel[1] = 1.0 if is_ahead(left) else 0.0
        food_rel[2] = 1.0 if is_ahead(right) else 0.0
        food_rel[3] = 1.0 if (not any(food_rel)) else 0.0

        dir_onehot = [0.0, 0.0, 0.0, 0.0]
        dir_onehot[DIR_INDEX[self.direction]] = 1.0

        def wall_dist(d):
            dx, dy = DIR_VEC[d]
            x, y = hx, hy
            dist = 0
            while True:
                x, y = x + dx, y + dy
                if x < 0 or x >= self.width or y < 0 or y >= self.height:
                    break
                dist += 1
            return dist / max(self.width, self.height)

        food_dist = (abs(hx - fx) + abs(hy - fy)) / (self.width + self.height)

        # 自由空间比例：剩余空格 / 总格数
        free_ratio = (self.max_cells - len(self.snake)) / self.max_cells

        # 头部被包围程度：蛇头周围一圈（含对角）被墙或身体占的比例
        trapped = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                x, y = hx + dx, hy + dy
                if x < 0 or x >= self.width or y < 0 or y >= self.height \
                        or (x, y) in body:
                    trapped += 1
        head_trap = trapped / 8.0

        return np.array(
            [danger(fwd), danger(left), danger(right)] + food_rel + dir_onehot
            + [wall_dist(fwd), wall_dist(left), wall_dist(right),
               food_dist, len(self.snake) / self.max_cells]
            + [safety_len(fwd), safety_len(left), safety_len(right),
               free_ratio, head_trap],
            dtype=np.float32,
        )

    def _reachable_free(self):
        """从蛇头出发计算可到达的空格数（衡量"还有多大活动空间"）。

        用 144 位整数做 flood fill：身体占位为 0，空格为 1，
        从蛇头位出发反复向四邻域扩散，直到不再变化。
        """
        occ = 0
        for x, y in self.snake[1:]:
            occ |= 1 << (y * self.width + x)
        free = ((1 << (self.width * self.height)) - 1) ^ occ
        hx, hy = self.snake[0]
        fill = free & (1 << (hy * self.width + hx))
        # 预先计算"跨行不溢出"的掩码：最右列不向左移，最左列不向右移
        right_mask = 0
        left_mask = 0
        for y in range(self.height):
            right_mask |= 1 << (y * self.width + (self.width - 1))
            left_mask |= 1 << (y * self.width + 0)
        while True:
            grow = fill
            grow |= (fill << self.width) & free
            grow |= (fill >> self.width) & free
            grow |= ((fill & ~right_mask) << 1) & free
            grow |= ((fill & ~left_mask) >> 1) & free
            if grow == fill:
                break
            fill = grow
        return bin(fill).count("1") - 1   # 减去蛇头自身，与"可达空格数"一致

    # ---------------------------------------------------------- 环境接口
    def reset(self):
        self._reset()
        return self._get_state()

    def step(self, action):
        """执行动作，返回 (next_state, reward, done, info)。"""
        assert not self.done, "episode already finished, call reset()"
        assert action in (0, 1, 2), "action must be 0/1/2"

        self.steps += 1
        self.direction = TURN[self.direction][action]
        dx, dy = DIR_VEC[self.direction]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy

        # 撞墙或撞身体
        hit_wall = nx < 0 or nx >= self.width or ny < 0 or ny >= self.height
        hit_body = (nx, ny) in self.snake

        # 移动（先去掉尾巴，如果没吃到食物）
        ate = (nx, ny) == self.food
        self.snake.insert(0, (nx, ny))
        if ate:
            self.food_eaten += 1
        else:
            self.snake.pop()

        reward = 0.0
        if hit_wall or hit_body:
            self.done = True
            reward = REWARD_HIT_WALL
        elif ate:
            self.done = False
            reward = REWARD_EAT
            self._place_food()
            # 吃到食物后蛇变长、食物换位，重新校准可达空间基准
            self._last_reach = self._reachable_free()
        else:
            # 每步小惩罚，催促尽快吃到食物
            reward = REWARD_STEP
            # shaping A：靠近食物奖励（potential-based）
            old_dist = abs(hx - self.food[0]) + abs(hy - self.food[1])
            new_dist = abs(nx - self.food[0]) + abs(ny - self.food[1])
            reward += SHAPING_FOOD_COEF * (old_dist - new_dist)
            # shaping B：保持/扩大可达空间（potential-based，防止钻死胡同）。
            # 只在蛇变长后启用——短蛇期活动空间巨大，此信号无意义且拖慢训练。
            if len(self.snake) >= SHAPING_REACH_MIN_LEN:
                # 只需算一次新状态的 potential，与上次基准相减即可。
                reach = self._reachable_free()
                reward += SHAPING_REACH_COEF * (reach - self._last_reach)
                self._last_reach = reach
            else:
                self._last_reach = self._reachable_free()

        if not self.done and self.steps >= self._step_limit():
            # 步数超限 = 绕圈失败，结束本局（不算"撞"所以给温和惩罚）
            self.done = True
            reward = -0.5

        self.last_reward = reward
        self.info = {
            "length": len(self.snake),
            "food_eaten": self.food_eaten,
            "steps": self.steps,
        }
        return self._get_state(), reward, self.done, self.info

    # ---------------------------------------------------------- 可视化
    def render_text(self):
        """终端字符画，便于调试与演示。"""
        board = [["·" for _ in range(self.width)] for _ in range(self.height)]
        for (x, y) in self.snake[1:]:
            board[y][x] = "█"
        hx, hy = self.snake[0]
        board[hy][hx] = "@"
        fx, fy = self.food
        board[fy][fx] = "★"
        return "\n".join(" ".join(row) for row in board)


if __name__ == "__main__":
    # 简单自测：随机策略跑 3 局
    env = SnakeEnv(seed=42)
    for ep in range(3):
        state = env.reset()
        total, steps = 0.0, 0
        while True:
            action = env.rng.randint(0, 2)
            state, r, done, info = env.step(action)
            total += r
            steps += 1
            if done:
                break
        print(f"随机策略 第{ep+1}局: 得分={total:+.2f} 步数={steps} 长度={info['length']}")
    print("环境自测通过")
