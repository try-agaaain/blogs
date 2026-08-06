"""bc_train.py 端到端测试：小规模训练出模型、非法数据被拒绝。"""

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestBcTrain(unittest.TestCase):
    def test_end_to_end(self):
        """小规模 BC 训练能产出可回读的 bc_policy.json。"""
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "bc_policy.json")
            data = os.path.join(td, "bc_data.npz")
            cmd = [sys.executable, os.path.join(_ROOT, "scripts", "bc_train.py"),
                   "--episodes", "3", "--max-steps", "500", "--epochs", "2",
                   "--seed", "1", "--out", out, "--data", data]
            r = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True,
                               timeout=600)
            self.assertEqual(r.returncode, 0,
                             f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}")
            self.assertTrue(os.path.exists(out), r.stdout + r.stderr)
            self.assertTrue(os.path.exists(data), "专家数据应被保存")
            # 模型可被 load 回读且结构正确
            from src.snake_game import STATE_DIM
            from src.ppo import BottleneckPolicy
            net = BottleneckPolicy(STATE_DIM, 3, hidden=56, n_blocks=3, seed=0)
            net.load(out)
            self.assertEqual(net.count_params(), 17868)

    def test_bad_data_rejected(self):
        """状态维度错误的 --data 文件应被拒绝（非零退出）。"""
        with tempfile.TemporaryDirectory() as td:
            bad = os.path.join(td, "bad.npz")
            np.savez(bad, states=np.zeros((5, 3), dtype=np.float32),
                     actions=np.zeros(5, dtype=np.int64))
            out = os.path.join(td, "bc_policy.json")
            cmd = [sys.executable, os.path.join(_ROOT, "scripts", "bc_train.py"),
                   "--data", bad, "--out", out]
            r = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True,
                               timeout=300)
            self.assertNotEqual(r.returncode, 0, "非法数据应报错")
            self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
