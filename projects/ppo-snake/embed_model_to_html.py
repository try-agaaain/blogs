"""把 ai_model.js 的 AI_MODEL 权重嵌入同目录下的 PPO贪吃蛇讲解.html。

用法（默认嵌入同目录的讲解页）：
    python embed_model_to_html.py [--html <PPO贪吃蛇讲解.html 的路径>]
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_JS = os.path.join(HERE, "ai_model.js")

parser = argparse.ArgumentParser()
parser.add_argument("--html", default=None,
                    help="PPO贪吃蛇讲解.html 的路径（默认取同目录）")
args = parser.parse_args()

HTML = args.html or os.path.join(HERE, "PPO贪吃蛇讲解.html")
if not os.path.exists(HTML):
    raise SystemExit(f"找不到 {HTML}，请用 --html 显式指定")

with open(MODEL_JS, encoding="utf-8") as f:
    model_js = f.read().strip()

with open(HTML, encoding="utf-8") as f:
    html = f.read()

anchor = "<script>/* 回放数据"
insert = f"""<script>/* 浏览器内实时运行的策略网络权重（由 export_model.py 从 best_snake.npz 导出，
   供「AI 自动玩」现场决策使用：每次点击都是一局新的实时推理，结果随随机性变化） */
{model_js}
</script>
{anchor}"""

assert anchor in html, "找不到锚点"
assert "var AI_MODEL" not in html, "AI_MODEL 已存在，避免重复嵌入"

html = html.replace(anchor, insert, 1)

with open(HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"已把 AI_MODEL（{len(model_js)//1024} KB）嵌入 {os.path.basename(HTML)}")
