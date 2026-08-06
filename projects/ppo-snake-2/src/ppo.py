"""ppo-snake-2 PPO 算法与策略网络。

网络：bottleneck 门控残差策略网络（与旧项目 sim_rel 实验一致的 19k 结构）：
  * 共享 trunk：in_proj → N 个 BottleneckSwiGLUResBlock
  * 双头：Actor (Linear → n_actions) / Critic (Linear → 1)

算法：PPO-Clip + GAE + 熵正则 + KL 早停（与旧项目已验证配置一致）。
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _BottleneckSwiGLUResBlock(nn.Module):
    """瓶颈 SwiGLU 残差块：h ← h + SiLU(up(gate ⊙ SiLU(down(h))))。

    每块参数量 ≈ 1.5H²，比全量 SwiGLU（2H²）省参数，可堆更深。
    """

    def __init__(self, dim):
        super().__init__()
        self.down = nn.Linear(dim, dim // 2)
        self.gate = nn.Linear(dim, dim // 2)
        self.up = nn.Linear(dim // 2, dim)
        for m in (self.down, self.gate, self.up):
            nn.init.orthogonal_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        d = self.down(x)
        g = self.gate(x)
        return x + self.up(d * F.silu(g))


class BottleneckPolicy(nn.Module):
    """bottleneck 门控策略网络：共享 trunk + 双头（~17k 参数）。"""

    def __init__(self, state_dim, n_actions, hidden=56, n_blocks=3, seed=0,
                 device=None):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.sd, self.na = state_dim, n_actions
        self.h, self.nb = hidden, n_blocks
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.in_proj = nn.Linear(state_dim, hidden)
        nn.init.orthogonal_(self.in_proj.weight, 1.0)
        nn.init.constant_(self.in_proj.bias, 0.0)
        self.blocks = nn.ModuleList(
            [_BottleneckSwiGLUResBlock(hidden) for _ in range(n_blocks)])
        self.actor = nn.Linear(hidden, n_actions)
        nn.init.orthogonal_(self.actor.weight, 0.1)
        nn.init.constant_(self.actor.bias, 0.0)
        self.critic = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.critic.weight, 0.5)
        nn.init.constant_(self.critic.bias, 0.0)

        self.to(self.device)

    def _trunk(self, x):
        h = F.silu(self.in_proj(x))
        for blk in self.blocks:
            h = blk(h)
        return h

    def forward(self, x):
        """x: (N, sd) -> (logits, value)。"""
        h = self._trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def _to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        x = np.asarray(x, dtype=np.float32)
        return torch.from_numpy(x).to(self.device)

    def _to_numpy(self, t):
        return t.detach().cpu().numpy()

    # ---------------- 前向接口 ----------------
    def act(self, x, eps=0.0, greedy=False):
        """给定状态返回 (动作, 对数概率, 价值, 概率分布)。

        greedy=True 取 argmax（logp 与动作严格对应）；
        否则按策略采样，eps>0 时 epsilon-greedy（以 eps 概率均匀随机）。
        输入单行向量返回 Python 标量动作。
        """
        single = (np.ndim(x) == 1)
        xt = self._to_tensor(np.atleast_2d(x))
        with torch.no_grad():
            logits, value = self.forward(xt)
            logp_all = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            if greedy:
                a = torch.argmax(logits, dim=-1)
                logp_a = logp_all.gather(1, a.unsqueeze(-1)).squeeze(-1)
                p_out = probs
            else:
                if eps > 0:
                    probs = (1 - eps) * probs + eps / self.na
                dist = torch.distributions.Categorical(probs=probs)
                a = dist.sample()
                logp_a = dist.log_prob(a)
                p_out = probs
        a_np = self._to_numpy(a)
        v_np = self._to_numpy(value)
        p_np = self._to_numpy(p_out)
        lp_np = self._to_numpy(logp_a)
        if single:
            return int(a_np[0]), float(lp_np[0]), float(v_np[0]), p_np[0]
        return a_np, lp_np, v_np, p_np

    def batch_eval(self, states):
        """批量返回 (logits, value, probs)。"""
        xt = self._to_tensor(states)
        with torch.no_grad():
            logits, value = self.forward(xt)
            probs = F.softmax(logits, dim=-1)
        return self._to_numpy(logits), self._to_numpy(value), \
            self._to_numpy(probs)

    def sample_actions(self, states):
        """批量按策略采样动作，返回 (actions, logp, value)。"""
        xt = self._to_tensor(states)
        with torch.no_grad():
            logits, value = self.forward(xt)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)
        return (self._to_numpy(a), self._to_numpy(logp), self._to_numpy(value))

    def greedy_actions(self, states):
        """批量贪心动作，返回 (actions, logp, value)，logp 与动作严格对应。

        与 sample_actions 同构，供评估/演示使用，避免 logp 与动作错配。
        """
        xt = self._to_tensor(states)
        with torch.no_grad():
            logits, value = self.forward(xt)
            a = torch.argmax(logits, dim=-1)
            logp_a = F.log_softmax(logits, dim=-1) \
                .gather(1, a.unsqueeze(-1)).squeeze(-1)
        return (self._to_numpy(a), self._to_numpy(logp_a), self._to_numpy(value))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    # ---------------- 序列化 ----------------
    def save(self, path):
        data = {
            "format": "ppo-snake-2-weights",
            "version": 1,
            "state_dim": self.sd,
            "n_actions": self.na,
            "hidden": self.h,
            "n_blocks": self.nb,
            "weights": {k: v.detach().cpu().numpy().tolist()
                        for k, v in self.state_dict().items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("format") == "ppo-snake-2-weights"
        self.load_state_dict({
            k: torch.tensor(v) for k, v in data["weights"].items()})
        self.to(self.device)
        return self


# ---------------------------------------------------------------- PPO
def compute_gae(rewards, dones, values, gamma=0.99, lam=0.95):
    """GAE 计算。

    values 长度为 T+1（values[t+1] 即 t 步末态 bootstrap value，
    由调用方填充 next_value）。dones[t] 为真时 bootstrap 归零。
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        adv[t] = last_gae
    ret = adv + values[:T]
    return adv.astype(np.float32), ret.astype(np.float32)


class PPOTrainer:
    """PPO-Clip 训练器（GAE + 熵正则 + KL 早停）。"""

    def __init__(self, policy, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
                 ent_coef=0.01, vf_coef=0.5, kl_threshold=0.05,
                 device=None):
        self.policy = policy
        if device is None:
            device = policy.device
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.kl_threshold = kl_threshold
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # ---------------- 数据收集 ----------------
    def collect_rollout(self, envs, max_steps=1024, greedy=False):
        """在向量环境上收集一条 rollout。

        返回 dict（全部为环境主序 (N, T, ...)，每条环境的轨迹沿时间轴连续）：
          states (N,T,38)、actions (N,T)、old_logp (N,T)、rewards (N,T)、
          dones (N,T)、values (N,T)、next_value (N,)。
        死亡环境在记录后立即 reset 续局；next_value 为 rollout 末步之后
        的状态价值（存活环境 bootstrap 用，死亡环境被 GAE 的 (1-done) 截断）。
        """
        states = envs.get_states()
        N = states.shape[0]
        sbuf = np.zeros((N, max_steps) + states.shape[1:], dtype=np.float32)
        abuf = np.zeros((N, max_steps), dtype=np.int32)
        lbuf = np.zeros((N, max_steps), dtype=np.float32)
        rbuf = np.zeros((N, max_steps), dtype=np.float32)
        dbuf = np.zeros((N, max_steps), dtype=bool)
        vbuf = np.zeros((N, max_steps), dtype=np.float32)

        for t in range(max_steps):
            sbuf[:, t] = states
            if greedy:
                acts, logp_a, vals = self.policy.greedy_actions(states)
            else:
                acts, logp_a, vals = self.policy.sample_actions(states)
            vbuf[:, t] = vals
            abuf[:, t] = acts
            lbuf[:, t] = logp_a
            states, rewards, dones, _ = envs.step(acts)
            rbuf[:, t] = rewards
            dbuf[:, t] = dones
            # 死亡环境立即 reset（记录后开新局）
            dead = np.where(dones)[0]
            if len(dead):
                states = envs.reset(dead)

        # 末态价值引导（对死亡环境被 GAE 的 (1-done) 截断，不影响）
        with torch.no_grad():
            _, next_value = self.policy.forward(
                self.policy._to_tensor(states))
            next_value = next_value.detach().cpu().numpy()

        return {
            "states": sbuf, "actions": abuf, "old_logp": lbuf,
            "rewards": rbuf, "dones": dbuf, "values": vbuf,
            "next_value": next_value,
        }

    # ---------------- 更新 ----------------
    def update(self, rollout, epochs=4, minibatch=512):
        """对 rollout 做多轮 minibatch PPO 更新。

        rollout 布局为环境主序 (N, T)：states (N,T,38)、actions (N,T) 等。
        优势/回报与 states 同序（环境主序展平），保证每个 (state, action)
        转移配对的 adv/ret 来自同一条轨迹。
        返回 (policy_loss, value_loss, entropy, kl, clip_frac)。
        """
        states = rollout["states"]
        actions = rollout["actions"]
        old_logp = rollout["old_logp"]
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        values = rollout["values"]
        next_value = rollout["next_value"]

        N, T = states.shape[0], states.shape[1]

        # GAE 按环境序列计算，输出 (N, T) 与 states 同序
        adv, ret = self._gae(rewards, dones, values, next_value)
        # 优势标准化
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 展平：所有数组共用同一 reshape（环境主序），索引天然对齐
        flat_s = states.reshape(N * T, -1)
        flat_a = actions.reshape(N * T)
        flat_lp = old_logp.reshape(N * T)
        flat_adv = adv.reshape(N * T)
        flat_r = ret.reshape(N * T)

        # 在线计算 value（旧 value 用于 GAE，新 value 用于 vf loss）
        n_updates = 0
        pl_sum = vl_sum = ent_sum = kl_sum = cf_sum = 0.0
        idx = np.arange(N * T)
        for _ in range(epochs):
            np.random.shuffle(idx)
            epoch_kl = 0.0
            epoch_updates = 0
            for start in range(0, N * T, minibatch):
                mb = idx[start:start + minibatch]
                s = self.policy._to_tensor(flat_s[mb])
                a = torch.from_numpy(flat_a[mb]).to(self.device)
                old_lp = torch.from_numpy(flat_lp[mb]).to(self.device)
                adv_mb = torch.from_numpy(flat_adv[mb]).to(self.device)
                ret_mb = torch.from_numpy(flat_r[mb]).to(self.device)

                logits, value = self.policy.forward(s)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(a)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - old_lp)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, ret_mb)

                loss = (policy_loss
                        + self.vf_coef * value_loss
                        - self.ent_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                with torch.no_grad():
                    kl = (old_lp - logp.detach()).mean().item()
                    clipped = ((ratio > 1 + self.clip) |
                               (ratio < 1 - self.clip)).float().mean().item()
                pl_sum += policy_loss.item()
                vl_sum += value_loss.item()
                ent_sum += entropy.item()
                kl_sum += kl
                cf_sum += clipped
                epoch_kl += kl
                n_updates += 1
                epoch_updates += 1

            # KL 早停：本 epoch 平均 KL 超阈值则提前终止内层循环
            if epoch_updates and (epoch_kl / epoch_updates) > self.kl_threshold:
                break

        return (pl_sum / n_updates, vl_sum / n_updates,
                ent_sum / n_updates, kl_sum / n_updates, cf_sum / n_updates)

    def _gae(self, rewards, dones, values, next_value):
        """按环境序列计算 GAE。

        输入均 (N, T)（values 为 rollout 旧价值，(N, T)），next_value (N,)。
        返回 (N, T) 的 (advantage, return)，与 states 同序。
        """
        N, T = rewards.shape
        adv = np.zeros((N, T), dtype=np.float32)
        ret = np.zeros((N, T), dtype=np.float32)
        for i in range(N):
            v_seq = np.concatenate([values[i], [next_value[i]]])
            adv[i], ret[i] = compute_gae(
                rewards[i], dones[i], v_seq, gamma=self.gamma, lam=self.lam)
        return adv, ret

    # ---------------- 评估 ----------------
    def evaluate(self, n_episodes=16, eps=0.0, greedy=True, max_steps=None):
        """贪心（默认）或 epsilon-greedy 评估，返回平均身长与食物数。

        用向量环境并行收集 n_episodes 局，每局在其死亡步记录终局信息；
        未在 max_steps 内结束的局（贪心策略可能陷入追尾循环）按当前长度
        截断记录。max_steps 默认 2000，足以判定蛇的生存能力。
        """
        from src.snake_vec import VectorSnakeEnv
        if max_steps is None:
            max_steps = 2000
        envs = VectorSnakeEnv(n_envs=n_episodes,
                              seeds=list(range(n_episodes)))
        states = envs.get_states()
        lens = np.zeros(n_episodes, dtype=np.int32)
        foods = np.zeros(n_episodes, dtype=np.int32)
        recorded = np.zeros(n_episodes, dtype=bool)
        for _ in range(max_steps):
            a, _, _, _ = self.policy.act(states, eps=eps, greedy=greedy)
            states, rewards, dones, infos = envs.step(a)
            newly = np.where(dones & ~recorded)[0]
            for i in newly:
                lens[i] = infos[i]["length"]
                foods[i] = infos[i]["food_eaten"]
                recorded[i] = True
            if recorded.all():
                break
            dead = np.where(dones)[0]
            if len(dead):
                states = envs.reset(dead)
        # 未在 max_steps 内结束的局，用当前长度/食物补足
        for i in np.where(~recorded)[0]:
            lens[i] = envs.lens[i]
            foods[i] = envs.food_eaten[i]
        return float(np.mean(lens)), float(np.mean(foods))
