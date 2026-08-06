"""ppo-snake-2 策略网络与 PPO 测试。"""

import os
import tempfile
import unittest

import numpy as np
import torch

from src.snake_game import STATE_DIM
from src.snake_vec import VectorSnakeEnv
from src.ppo import BottleneckPolicy, PPOTrainer, compute_gae


class TestPolicy(unittest.TestCase):
    def test_param_count(self):
        """参数 < 20k（用户约束）。"""
        net = BottleneckPolicy(STATE_DIM, 3, hidden=56, n_blocks=3, seed=0)
        self.assertLess(net.count_params(), 20000)
        # FRONT_SCAN=6（56 维状态）时 hidden=56, blocks=3 的参数数
        self.assertEqual(net.count_params(), 17868)

    def test_forward_shape(self):
        net = BottleneckPolicy(STATE_DIM, 3, hidden=56, n_blocks=3, seed=0)
        x = np.random.rand(8, STATE_DIM).astype(np.float32)
        logits, value = net.forward(net._to_tensor(x))
        self.assertEqual(logits.shape, (8, 3))
        self.assertEqual(value.shape, (8,))

    def test_act_single(self):
        net = BottleneckPolicy(STATE_DIM, 3, hidden=56, n_blocks=3, seed=0)
        s = np.random.rand(STATE_DIM).astype(np.float32)
        a, logp, v, p = net.act(s)
        self.assertIn(a, (0, 1, 2))
        self.assertTrue(np.isclose(np.sum(p), 1.0, atol=1e-5))

    def test_act_greedy_matches_argmax(self):
        """greedy=True 时动作=argmax，logp 与动作严格对应。"""
        net = BottleneckPolicy(STATE_DIM, 3, hidden=56, n_blocks=3, seed=0)
        s = np.random.rand(STATE_DIM).astype(np.float32)
        a, logp, _, _ = net.act(s, greedy=True)
        logits, _, probs = net.batch_eval(s[None])
        self.assertEqual(a, int(np.argmax(logits[0])))
        self.assertAlmostEqual(logp, float(np.log(probs[0, a])), places=5)

    def test_act_eps_probs_mixed(self):
        """eps>0 时返回的 probs 必须包含均匀混合（与采样分布一致）。"""
        net = BottleneckPolicy(STATE_DIM, 3, hidden=56, n_blocks=3, seed=0)
        s = np.random.rand(STATE_DIM).astype(np.float32)
        eps = 0.2
        _, _, _, p = net.act(s, eps=eps)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=6)
        self.assertTrue(np.all(p >= eps / 3 - 1e-6))

    def test_save_load(self):
        net = BottleneckPolicy(STATE_DIM, 3, hidden=56, n_blocks=3, seed=1)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "net.json")
            net.save(path)
            net2 = BottleneckPolicy(STATE_DIM, 3, hidden=56, n_blocks=3,
                                    seed=2)
            net2.load(path)
            x = np.random.rand(4, STATE_DIM).astype(np.float32)
            l1, v1 = net.forward(net._to_tensor(x))
            l2, v2 = net2.forward(net2._to_tensor(x))
            self.assertTrue(np.allclose(
                l1.detach().cpu().numpy(), l2.detach().cpu().numpy(),
                atol=1e-6))


class TestPPO(unittest.TestCase):
    def test_smoke_training(self):
        """小规模冒烟：PPO rollout + update 不报错且 loss 有限。"""
        net = BottleneckPolicy(STATE_DIM, 3, hidden=56, n_blocks=3, seed=0)
        trainer = PPOTrainer(net, lr=3e-4)
        envs = VectorSnakeEnv(n_envs=8, seeds=list(range(8)))
        rollout = trainer.collect_rollout(envs, max_steps=32)
        # rollout 为环境主序 (N, T)
        self.assertEqual(rollout["states"].shape, (8, 32, STATE_DIM))
        self.assertEqual(rollout["actions"].shape, (8, 32))
        self.assertEqual(rollout["rewards"].shape, (8, 32))
        self.assertEqual(rollout["dones"].shape, (8, 32))
        self.assertEqual(rollout["next_value"].shape, (8,))
        pl, vl, ent, kl, clipf = trainer.update(rollout, epochs=2,
                                                minibatch=64)
        self.assertTrue(np.isfinite(pl))
        self.assertTrue(np.isfinite(vl))
        self.assertTrue(ent > 0)

    def test_evaluate(self):
        """评估返回身长/食物均值。"""
        net = BottleneckPolicy(STATE_DIM, 3, hidden=56, n_blocks=3, seed=0)
        trainer = PPOTrainer(net)
        avg_len, avg_food = trainer.evaluate(n_episodes=4)
        self.assertGreaterEqual(avg_len, 0)
        self.assertGreaterEqual(avg_food, 0)

    def test_greedy_rollout_logp_matches(self):
        """greedy rollout：动作=argmax，old_logp 与动作严格对应，且填充 value。"""
        net = BottleneckPolicy(STATE_DIM, 3, hidden=56, n_blocks=3, seed=0)
        trainer = PPOTrainer(net)
        envs = VectorSnakeEnv(n_envs=4, seeds=list(range(4)))
        ro = trainer.collect_rollout(envs, max_steps=16, greedy=True)
        states = ro["states"].reshape(-1, STATE_DIM)
        logits, _, _ = net.batch_eval(states)
        argmax_a = np.argmax(logits, axis=1)
        self.assertTrue(np.array_equal(ro["actions"].ravel(), argmax_a))
        # old_logp 必须等于 argmax 动作的 log_softmax
        logp_all = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        logp_argmax = logp_all[np.arange(len(states)), argmax_a]
        self.assertTrue(np.allclose(ro["old_logp"].ravel(), logp_argmax,
                                    atol=1e-5))
        # greedy rollout 也应有价值估计（可被 GAE 使用）
        self.assertTrue(np.any(np.abs(ro["values"]) > 0))


class TestGAE(unittest.TestCase):
    """GAE 精确值、done 截断、逐环境对齐与展平顺序回归。"""

    def test_compute_gae_exact(self):
        """γ=0.99 λ=0.95 的精确值手算验证。"""
        rewards = np.array([0.0, 1.0, 0.0])
        dones = np.array([False, False, True])
        values = np.zeros(4)
        adv, ret = compute_gae(rewards, dones, values)
        # t=2(done): gae=0；t=1: 1；t=0: 0.99*0.95*1
        self.assertAlmostEqual(adv[2], 0.0, places=5)
        self.assertAlmostEqual(adv[1], 1.0, places=5)
        self.assertAlmostEqual(adv[0], 0.99 * 0.95, places=5)
        # ret = adv + values[:T]
        self.assertTrue(np.allclose(ret, adv))

    def test_compute_gae_truncates_bootstrap_at_done(self):
        """done 截断 bootstrap：done 处的未来价值不计入。"""
        rewards = np.array([0.0, 0.0, 1.0])
        dones = np.array([False, False, True])
        values = np.array([0.0, 0.0, 0.0, 10.0])
        adv_done, _ = compute_gae(rewards, dones, values)
        # 截断：adv[2] 只含即时奖励 1，而非 1 + 0.99*10
        self.assertAlmostEqual(adv_done[2], 1.0, places=5)
        # 对照：不截断时 bootstrap 10 会进入
        adv_open, _ = compute_gae(rewards, np.zeros_like(dones, dtype=bool),
                                  values)
        self.assertAlmostEqual(adv_open[2], 1.0 + 0.99 * 10.0, places=5)

    def test_gae_flat_aligns_with_states(self):
        """回归：GAE 展平顺序必须与 states 展平顺序一致。

        原 bug 是 rollout 按 (T, N) 存、GAE 按 (N, T) 算后直接 reshape，
        导致每个 state-action 配对到错误环境/时刻的优势。现在统一
        (N, T) 环境主序，此处用 states 编码位置验证逐项对齐。
        """
        net = BottleneckPolicy(STATE_DIM, 3, hidden=56, n_blocks=3, seed=0)
        trainer = PPOTrainer(net)
        N, T = 2, 5
        # states 末位编码 (i*T+t)，供事后核对配对
        states = np.zeros((N, T, STATE_DIM), dtype=np.float32)
        states[:, :, -1] = np.arange(N * T).reshape(N, T)
        rewards = np.zeros((N, T))
        rewards[0, 2] = 1.0            # 仅 env0 t=2 有正奖励
        dones = np.zeros((N, T), dtype=bool)
        dones[1, 3] = True
        values = np.zeros((N, T))
        next_value = np.zeros(N)

        adv, ret = trainer._gae(rewards, dones, values, next_value)
        self.assertEqual(adv.shape, (N, T))
        # 每环境与单序列 compute_gae 一致
        for i in range(N):
            v_seq = np.concatenate([values[i], [next_value[i]]])
            exp_adv, _ = compute_gae(rewards[i], dones[i], v_seq,
                                     gamma=trainer.gamma, lam=trainer.lam)
            self.assertTrue(np.allclose(adv[i], exp_adv))
        # 展平后第 k 项必须落在 (i=k//T, t=k%T)，与 states 编码一致
        states_flat = states.reshape(N * T, -1)
        adv_flat = adv.reshape(N * T)
        for k in range(N * T):
            i, t = k // T, k % T
            self.assertEqual(int(states_flat[k, -1]), k)
            self.assertAlmostEqual(float(adv_flat[k]),
                                   float(adv[i, t]), places=5)

    def test_update_advantage_orienting(self):
        """端到端：正优势样本的 logp 变化应显著高于零/负优势样本。

        构造 N=1 的确定性 rollout（优势可精确手算），update 后正优势的
        (state, action) 应相对被提升（回归错位 bug：若 adv 错配，该差异
        不会出现在对应位置）。
        """
        net = BottleneckPolicy(STATE_DIM, 3, hidden=56, n_blocks=3, seed=0)
        trainer = PPOTrainer(net, lr=1e-2)
        N, T = 1, 8
        rng = np.random.RandomState(0)
        states = rng.rand(N, T, STATE_DIM).astype(np.float32)
        logits, _, _ = net.batch_eval(states.reshape(-1, STATE_DIM))
        actions = np.array([logits[k].argmax() for k in range(N * T)],
                           dtype=np.int32).reshape(N, T)
        # old_logp：用当前策略的 argmax 动作 logp
        lp_all = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        old_logp = lp_all[np.arange(N * T), actions.reshape(-1)] \
            .astype(np.float32).reshape(N, T)
        rewards = np.zeros((N, T))
        rewards[0, 2] = 1.0            # 唯一正奖励
        dones = np.zeros((N, T), dtype=bool)
        values = np.zeros((N, T))
        next_value = np.zeros(N)

        adv, _ = trainer._gae(rewards, dones, values, next_value)
        pos = adv.reshape(-1) > 0
        self.assertTrue(pos.any())

        rollout = {
            "states": states, "actions": actions, "old_logp": old_logp,
            "rewards": rewards, "dones": dones, "values": values,
            "next_value": next_value,
        }
        at = torch.from_numpy(actions.reshape(-1)).to(net.device)
        with torch.no_grad():
            logits0, _ = net.forward(net._to_tensor(
                states.reshape(-1, STATE_DIM)))
            lp0 = torch.distributions.Categorical(logits=logits0) \
                .log_prob(at).cpu().numpy()
        trainer.update(rollout, epochs=1, minibatch=N * T)
        with torch.no_grad():
            logits1, _ = net.forward(net._to_tensor(
                states.reshape(-1, STATE_DIM)))
            lp1 = torch.distributions.Categorical(logits=logits1) \
                .log_prob(at).cpu().numpy()
        d = lp1 - lp0
        # 正优势样本相对提升幅度须大于零/负优势样本
        self.assertGreater(d[pos].mean(), d[~pos].mean())


if __name__ == "__main__":
    unittest.main()
