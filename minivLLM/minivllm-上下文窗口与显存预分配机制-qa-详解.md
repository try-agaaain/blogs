# MinivLLM 上下文窗口与显存预分配机制 · Q&A 详解

> 本文系统梳理 MinivLLM 中 `max_position` 与 `max_model_length` 两个长度限制的含义与区别、系统启动时的内存预分配流程（warmup → KV cache 池 → CUDA graph）、以及为什么使用 prefill 的峰值而不是 decode 来做内存基准。

---

## Q1 · `max_position` 和 `max_model_length` 分别是什么？它们的区别在哪？

两者都是长度上限，但作用的层面和"硬度"完全不同：

| | `max_position` (32768) | `max_model_length` (128) |
|---|---|---|
| **作用层** | 模型结构层 — RotaryEmbedding | 推理调度层 — Scheduler |
| **定义位置** | `main.py:35` config → `rotary_embedding.py:66` | `main.py:38` config → `sampling_parameters.py:9` → 每请求可配 |
| **定死时机** | 模型初始化时，预计算 `cos_sin_cache` 并存为 `register_buffer`，不可变 | 每次请求由 `SamplingParams` 传入，运行时可变 |
| **约束方式** | `rotary_embedding.py:104` 从 cache 按 token 位置索引取值，超出 `max_position` 直接越界报错 | `scheduler.py:87` 每 step 检查 `seq.num_tokens >= seq.max_model_length`，超出则标记 FINISHED 并释放 KV cache 块 |
| **语义** | "模型能编码的最大 token 位置索引"（硬上限，硬件级） | "单次请求允许的最大总 token 数，含 prompt + 生成"（软上限，策略级） |

用一句话概括：`max_position` 是模型能"看见"多远的极限，`max_model_length` 是你允许它看到多远。

**为什么 Qwen3 的 `max_position` 是 32768，而 `max_model_length` 只设了 128？**

因为这是一个**本地测试/调试配置**。`max_position=32768` 是模型出厂规格（Rotary Embedding 的 `cos_sin_cache` 就这么大，约 32768 × 128 × 4 = 16MB，固定开销）。`max_model_length=128` 是为了确保短 prompt 迅速跑通推理管线，避免 OOM。在生产环境中你可以把它调到 4096、8192 甚至 32768——只要显存装得下对应的 KV cache 块数。

---

## Q2 · `max_position` 的 32768 是在模型初始化时就定死无法更改的吗？

**是的。** 原因在于 `RotaryEmbedding.__init__` (`rotary_embedding.py:89-98`) 在构造时一次性预计算了整个位置空间的正余弦值：

```python
positions = torch.arange(self.max_position).float()          # (32768,)
freqs = torch.einsum("i,j -> ij", positions, self.inv_freq)  # (32768, 64)
cos_sin_cache = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)  # (32768, 128)
self.register_buffer("cos_sin_cache", cos_sin_cache)          # 注册为 buffer，不可训练
```

`register_buffer` 将这个 tensor 绑定到模型的状态字典中（随模型保存/加载），但梯度为 False。它就是一个**预计算的常量查找表**。前向传播时 (`rotary_embedding.py:104`)：

```python
cos_sin = self.cos_sin_cache[positions]  # 按 token 位置索引
```

请求中如果有 token 的位置 >= 32768，索引越界就直接报错。要扩大它，唯一的方法是**重建模型并重新传入更大的 `max_position`**。

---

## Q3 · 系统启动时是如何进行预分配的？完整流程是什么样的？

启动链路为 `LLMEngine.__init__` → `ModelRunner.__init__`。分三步进行：

### 第一步：Warmup 模型 (`model_runner.py:186-193`)

目的不是"预热"，而是**让 PyTorch 记录一次最大规模的 forward pass 所消耗的峰值显存**，后续据此计算 KV cache 池能分多大。

```python
max_tokens = config['max_num_batch_tokens']    # 4096
max_model_length = config['max_model_length']  # 128
batch_size = max_tokens // max_model_length    # 4096/128 = 32 条序列

# 构造 32 条各含 128 个零 token 的假序列，全部 prefill
seqs = [Sequence(token_ids=[0]*128, block_size=256) for _ in range(32)]
self.run(seqs, is_prefill=True)  # 跑一次完整 prefill forward

torch.cuda.empty_cache()  # 释放临时中间激活
```

此时 PyTorch 内部分配器记录了 `allocated_bytes.all.peak`，即这次 prefill 中达到的最大分配量。

### 第二步：分配 KV Cache 池 (`model_runner.py:197-253`)

根据 warmup 后的数值，**动态计算**实际能用的 KV cache 块数：

```python
free_mem, total_mem = torch.cuda.mem_get_info()   # GPU 当前空闲显存
total_free_mem = free_mem * 0.9                     # 保守点，只用 90%

peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak']      # warmup 峰值
current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current'] # 稳态（主要是权重）

# 核心公式：可用内存 = 90%空闲 - (峰值 - 稳态)，即去掉 prefill 必须的临时内存
available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)

# 每个 block 占多少字节
block_bytes = block_size * 2 * num_layers * num_kv_heads * head_dim * dtype_itemsize
            = 256 * 2 * 28 * 8 * 128 * 2 = 29,360,128 字节 ≈ 28 MB

num_available_kv_blocks = int(available_mem // block_bytes)
```

多 GPU 场景下，所有 rank 各自独立计算，再用 `dist.all_reduce(ReduceOp.MIN)` 取各卡的最小值作为全局统一上限 — 因为 KV cache 的整体容量受**最显存紧张的 GPU** 限制（`model_runner.py:224-242`）。

最终分配一个**全局大张量**作为 KV cache 池：

```python
allocated_kv_cache = torch.zeros(
    2,                    # K 和 V 各一套
    num_layers,           # 28 层
    max_cached_blocks,    # 动态算出的 block 总数
    block_size,           # 256 token/block
    num_kv_heads,         # 8 个 KV 头
    head_dim              # 128 维
)

# 把每层的 Attention 模块绑定到池子的对应层切片
layer_id = 0
for module in model.modules():
    if hasattr(module, 'k_cache'):
        module.k_cache = allocated_kv_cache[0, layer_id]  # 第 layer_id 层的 K 缓存
        module.v_cache = allocated_kv_cache[1, layer_id]  # 第 layer_id 层的 V 缓存
        layer_id += 1
```

这就是 PagedAttention 的核心：**一个预分配的全局池，按 `block_size`（256 token）粒度分块调度给各序列**。后续不再分配新的 GPU 内存 —— 所有序列的 K、V 都写入这个池的特定槽位。

### 第三步：捕获 CUDA Graph (`model_runner.py:406-457`)

为 decode 阶段（每步只生成 1 个 token）预捕获 CUDA Graph。所有输入/输出的 buffer 按**最坏情况**预分配：

```python
max_bs = config['max_num_seqs']           # 16 条并发序列
max_len = config['max_model_length']      # 128
max_num_blocks = ceil(max_len / block_size)  # ceil(128/256) = 1

input_ids = torch.zeros(max_bs, ...)         # (16,)
slot_mapping = torch.zeros(max_bs, ...)      # (16,)
context_lens = torch.zeros(max_bs, ...)      # (16,)
block_tables = torch.zeros(max_bs, 1, ...)   # (16, 1)
outputs = torch.zeros(max_bs, vocab_size, ...)
```

然后对常见 batch size [1, 2, 4, 8, 16] 分别捕获 CUDA Graph，decode 阶段按实际 batch size 选取匹配的 graph 做 replay。

---

## Q4 · GPU 显存是分成两块——一块预留模型、一块做 KV 缓存吗？

**不是静态的两块，而是运行时动态计算的逻辑分隔。** 整个 GPU 显存的占用结构如下：

```
┌───────────────────────────────────┐
│         模型权重 (persistent)       │  ← 加载后始终在 GPU 上
│         Q/K/V/O/MLP/Embedding 等   │     Qwen3-0.6B 约 1.2GB
├───────────────────────────────────┤
│         KV Cache 池 (persistent)    │  ← torch.zeros 一次性分配，常驻
│         形状: (2,28,N,256,8,128)   │     所有序列共享这个池
├───────────────────────────────────┤
│   prefill 时释放的临时激活          │  ← 只在 forward 期间存在
│   (大 QKV 张量、flash attn buffer) │     前向结束后由 allocator 回收
├───────────────────────────────────┤
│          安全 buffer               │  ← 留给 CUDA context / 碎片
└───────────────────────────────────┘
```

KV cache 池之所以不能把全部空闲显存都占满，是因为它必须与 prefill 的峰值临时内存**共存于同一时刻**。如果池太大，prefill 时就会 OOM。等式：

```
KV 池 ≤ 总显存 - 模型权重 - prefill峰值临时内存 - buffer
       = 空闲 × 0.9 - (peak - current)        # 即 Q3 中的公式
```

其中 `peak - current` 就是 prefill 比稳态多占用的那一截临时内存。

---

## Q5 · 为什么用 prefill 的峰值而不是 decode 的峰值来当模型的"预留内存"？

因为 **prefill 的内存压力远大于 decode**：

| | Prefill | Decode |
|---|---|---|
| 每序列处理的 token 数 | 整条 prompt（如 128 个） | 1 个 |
| Q 张量大小 | `(B × seq_len, num_heads, head_dim)` | `(B, 1, num_heads, head_dim)` |
| K、V 张量大小 | 同上 | `(B, num_kv_heads, head_dim)` |
| 注意力机制 | Flash Attention，QK^T 矩阵 `seq_len × seq_len` | Paged Attention，只算 1 行 |
| 临时内存 | **大**（全部激活值并行计算） | **极小**（单 token 激活值） |

如果以 decode 的微薄临时内存做基准，算出的 KV cache 池会过大。当真正执行 prefill 时，大池 + 大激活 同时存在，总显存超过物理上限，OOM。

**"用 prefill 峰值做基准"实际上是在说：KV cache 池的大小，必须保证在内存压力最大的时刻（prefill）也不越界。** decode 的内存压力小，自然也就安全。

用一个比喻：一个大货车（prefill 峰值内存）和一个自行车（decode 临时内存）要从同一扇门（总显存）过。你必须按大货车的体积来算还能放多大的柜子（KV 池），而不是按自行车。

---

## Q6 · warmup 为什么用 prefill 而不用 decode？这不是会导致短对话浪费部分显存吗？

**warmup 用 prefill 是为了测峰值，不是因为 prefill 是"常态"。**

warmup 的目的就是制造最极端的内存压力，让 PyTorch allocator 记录下 `peak_mem`。这个峰值用在哪里呢？——用在 Q3 的公式中**给 KV cache 池留出安全边际**：

```python
available_mem = total_free_mem - (peak_mem - current_mem)
```

`peak_mem - current_mem` 就是 prefill 比稳态多占的内存。"减去它"意味着 KV 池的大小确保了 prefill 时不会 OOM。

**短对话浪费吗？** 不完全是。KV cache 池是**固定容量**的公共资源，被多条并发序列按需瓜分。一个短对话可能只占用 1 个 block（256 token），但池子的剩余 block 可以同时服务其他序列。池子越大 = 可同时容纳的上下文总 token 数越多 = 吞吐越高。池子大小的本质是**空间换吞吐**，不是为每个请求量身定制。

---

## Q7 · 整个模型中，各层的 K、V 权重大小都是相同的吗？

**完全相同。** 所有 28 层共享同一套结构参数（`qwen3.py:257-271`）：

```python
hidden_size = 1024
num_kv_heads = 8
head_dim = 128

# 每层的 K 投影权重形状
K_weight: (num_kv_heads × head_dim) × hidden_size = (8 × 128) × 1024 = (1024, 1024)

# 每层的 V 投影权重形状
V_weight: (num_kv_heads × head_dim) × hidden_size = (8 × 128) × 1024 = (1024, 1024)
```

28 层完全一致，没有层间差异。所以在 `allocate_kv_cache` 中，每一层被分配到的 KV cache 切片大小也完全相同——池子是规整的矩形。

```
allocated_kv_cache 形状示意:

[K, 0]: ████████████████████████  ← 第 0 层 K cache: (max_blocks, 256, 8, 128)
[K, 1]: ████████████████████████  ← 第 1 层 K cache: 完全一样大
 ...                             ← ...
[K,27]: ████████████████████████  ← 第 27 层 K cache
[V, 0]: ████████████████████████  ← 第 0 层 V cache
 ...                             ← ...
[V,27]: ████████████████████████  ← 第 27 层 V cache
```

---

## Q8 · 启动预分配的完整参数链是什么样的？

所有相关参数及其作用可以画成一张依赖图：

```
main.py config:
  │
  ├── max_position = 32768 ──────→ RotaryEmbedding.cos_sin_cache 大小 (16MB, 不参与预分配公式)
  │
  ├── max_model_length = 128 ────→ ① warmup 序列长度
  │                               ② CUDA graph block_table 列数 = ceil(128/256) = 1
  │                               ③ 运行时停止条件
  │
  ├── block_size = 256 ─────────→ ① 每块 token 数
  │                               ② block_bytes 计算公式中的因子
  │                               ③ block_table 中 ceil() 的分母
  │
  ├── max_num_batch_tokens = 4096 → warmup batch 大小 = 4096/128 = 32
  │
  ├── max_num_sequences = 16 ───→ CUDA graph 最大 batch、block_table 行数
  │
  ├── gpu_memory_utilization = 0.9 → KV cache 池占空闲内存的比例
  │
  └── (num_layers=28, num_kv_heads=8, head_dim=128) → block_bytes 计算
```

注意 `max_position=32768` **不参与**预分配公式。它只在模型构造时决定了 `cos_sin_cache` 的 16MB 内存，与后续的 warmup、KV 池、CUDA graph 都无关。
