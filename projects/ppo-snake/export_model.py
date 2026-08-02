"""
把训练好的 best_snake.npz 权重导出为浏览器可用的 JS 文件（ai_model.js）。
供 HTML 页面的「AI 自动玩」实时决策使用——每次点击都由浏览器内运行的
同一套神经网络现场计算动作，结果随随机性变化，不再是固定回放。

用法：
    python export_model.py
输出：
    ai_model.js  —— const AI_MODEL = { W1:[...], b1:[...], W2p:[...], b2p:[...] }
"""

import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "checkpoint", "best_snake.npz")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_model.js")


def fmt_2d(a):
    """把 2D 数组展平为 JS 数字数组文本。用 repr 完整保留 float32 精度。"""
    return "[" + ",".join(repr(float(v)) for v in a.ravel()) + "]"


def main():
    assert os.path.exists(SRC), f"找不到模型 {SRC}，请先训练"
    d = np.load(SRC)

    # 策略网络参数：W1 (21,64) b1 (64,) W2p (64,3) b2p (3,)
    W1, b1 = d["W1"], d["b1"]
    W2p, b2p = d["W2p"], d["b2p"]

    js = f"""var AI_MODEL = {{
  h: {W1.shape[1]},
  na: {W2p.shape[1]},
  W1: {fmt_2d(W1)},
  b1: {fmt_2d(b1)},
  W2p: {fmt_2d(W2p)},
  b2p: {fmt_2d(b2p)}
}};
"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(js)
    n_params = W1.size + b1.size + W2p.size + b2p.size
    print(f"已导出策略网络权重 -> {OUT}")
    print(f"  维度: {W1.shape} -> {W2p.shape}，参数量 {n_params}，文件 {len(js)//1024} KB")


if __name__ == "__main__":
    main()
