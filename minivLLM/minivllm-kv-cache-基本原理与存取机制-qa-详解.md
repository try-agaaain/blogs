# MinivLLM KV Cache 基本原理与存取机制 · Q&A 详解

> 本文系统梳理 MinivLLM 中 KV Cache 的核心概念：什么是 KV Cache、它为什么能加速推理、每个 block 存的是什么、prefill 与 decode 阶段的 cache 操作有何不同、以及 cache 如何在 decode 时被分块读取。

---

## Q1 · 如果没有 KV Cache，大模型生成一个 token 是如何工作的？

答案是：**每生成一个新 token，要把历史上所有 token 的 K 和 V 全部重新算一遍。**

```
生成 token 128: input = [t0, t1, ..., t127] → 28 层 forward
  → 每层计算 128 组 Q、K、V → Attention(Q, K, V) → 出 token 128

生成 token 129: input = [t0, t1, ..., t128] → 28 层 forward
  → 每层计算 129 组 Q、K、V → Attention(Q, K, V) → 出 token 129

生成 token 130: input = [t0, t1, ..., t129] → 28 层 forward
  → 每层计算 130 组 Q、K、V → Attention(Q, K, V) → 出 token 130
```

每一步都在重复计算之前所有 token 的 K、V。总计算量呈 **O(N²)** 增长，内存也线性增长。生成越长，系统越慢，直至卡死。

---

## Q2 · 有 KV Cache 时，生成 token 的流程是什么？

KV Cache 的核心思路是：**每个 token 的 K 和 V 只计算一次，后续反复读取。**

```
═══════ 阶段一：Prefill（预填充）═══════
输入: 完整 prompt [t0, t1, ..., t127] 的 token IDs (1D, 128 个)

28 层 forward，每层:
  x → QKV投影 → 获得 Q(128,16,128), K(128,8,128), V(128,8,128)
  → RoPE(Q,K) → Flash Attention(Q,K,V) → 输出
  → store_kvcache: 把 K 和 V 写入 KV cache 池

lm_head → logits(128, vocab) → 取最后一个位置的 logit → sample → token 128

═══════ 阶段二：Decode（逐 token 生成）═══════
输入: 仅 [token 128] (1D, 1 个 token)

28 层 forward，每层:
  x → QKV投影 → 新算 Q(1,16,128), K(1,8,128), V(1,8,128)
  → RoPE → store_kvcache: 把新的 K,V 追加写入 cache
  → Paged Attention: Q × (cache 中全部历史 K) → 权重 × (cache 中全部历史 V) → 输出

lm_head → logits(1, vocab) → sample → token 129

接下来 token 130, 131, ... 都复用上述 decode 流程。
```

总结：prefill 建立初始 KV cache，decode 每次只算 1 个 token 的 K/V 并追加进 cache，然后从 cache 读取全部历史 K/V 完成注意力计算。这样**每个 token 的 K、V 恰好被算一次**，总计算量降为 O(N)。

---

## Q3 · KV Cache 中一个 block 存放的是什么？是权重吗？

**不是权重，是激活值（activations）。**

模型权重（W_k, W_v）是 `nn.Parameter`，存储在模型参数中，参与每次 forward 的 QKV 投影计算。KV cache 中的 block 存的是每次 forward **计算出的结果**——即投影后的 key 向量和 value 向量。

一个 block 的维度为：

```
(block_size=256, num_kv_heads=8, head_dim=128)
```

即 256 个 token × 8 个 KV 头 × 128 维 = 262,144 个 float 数（约 0.5 MB，fp16）。

代码验证：`attention.py:507` 的 `store_kvcache` 函数在每层 forward 结束时调用，把刚算出的 K、V 张量写入 cache 池中对应 slot。

```
可以这样理解：
├── 模型参数 (权重)
│   └── qkv_projection.weight:  把 hidden_states 映射为 Q/K/V 的矩阵（存的是学习到的"规则"）
├── KV Cache (激活值)
│   └── block[i]: 某 256 个 token 经过 qkv_projection.weight 计算后的实际 K 向量和 V 向量
│                  （存的是"计算结果"）
```

---

## Q4 · 每生成一个新的 token，需要缓存多少 K 和 V？

**全部 28 层，每层各一组 K 和一组 V。**

一个新 token 从第 0 层进入，逐层向上，每一层都做以下操作：

```
Layer 0: x → qkv_proj(x) → 得到 K(1,8,128), V(1,8,128) → 存入 Layer 0 的 cache 切片
Layer 1: x → qkv_proj(x) → 得到 K(1,8,128), V(1,8,128) → 存入 Layer 1 的 cache 切片
...
Layer 27: 同上 → 存入 Layer 27 的 cache 切片
```

这解释了 KV cache 池的形状 `(2, 28, max_blocks, block_size, 8, 128)`：第 0 维区分 K 和 V，第 1 维区分 28 层，每层有自己独立的 cache 空间。不同层的 K、V 绝不混用——第 3 层算出来的 K 只被第 3 层在后续 decode 中查询。

---

## Q5 · Prefill 和 Decode 阶段在存放 K、V 到 cache 时有差别吗？

**底层写入操作相同，都是 `store_kvcache`。差别在于写入的模式和位置计算方式。**

### Prefill 的写入模式

一次性写入多个 token，可能跨越多个连续的 block：

```python
# model_runner.py:288-293
for i, block_id in enumerate(seq.block_table[num_cached_blocks:]):
    if 不是最后一个块:
        slot_mappings.extend(range(block_id*256, (block_id+1)*256))   # 填满整块, 256 个 slot
    else:
        slot_mappings.extend(range(block_id*256, block_id*256 + 剩余token数))  # 尾部部分填充
```

例如 prompt 有 300 个 token，会分配 2 个 block（block_size=256）。第一个 block 分配 256 个连续 slot，第二个分配 44 个 slot。

### Decode 的写入模式

每次只写一个 slot，紧接在当前序列最后一块的最后一个有效 token 之后：

```python
# model_runner.py:326
slot_mapping = block_table[-1] * 256 + last_block_num_tokens - 1
```

| | Prefill | Decode |
|---|---|---|
| 一次写入量 | 整条 prompt（如 128 token） | 每序列 1 个 token |
| slot 分配 | 批量连续分配 | 单点追加 |
| 是否创建新块 | 可能创建多个 | 只有当前块满时才创建 1 个 |

---

## Q6 · Decode 时新 K、V 存放在 cache 的哪个位置？有规则吗？

**有规则：按序列内顺序追加，但物理 block 号可以是任意的。**

序列维护了一个 `block_table`，记录该序列占用了哪些物理块：

```
block_table = [物理块7, 物理块3, 物理块12]
```

当前 520 个 token（256+256+8）：

- token 0~255 在物理块 7
- token 256~511 在物理块 3
- token 512~519 在物理块 12（offset 0~7）

新来的第 521 个 token 写入 `物理块12 的 offset=8`。

当 offset 到达 256（即下一个 token 需要新块）：

- `block_manager.append()` 从 free list 弹出一个物理块（如 物理块 9）
- 追加到 `block_table` 末尾：`[7, 3, 12, 9]`
- 新块从头（offset=0）开始写入

**规则是"顺序追加"，物理块号是 free list 决定的——不同序列的 block 可以散落在池中任意位置。**

---

## Q7 · 为什么 cache 用 `i/256` 和 `i%256` 来定位？

这是 **逻辑位置到物理位置的映射**。cache 池按 block（每块 256 token）组织，而注意力计算需要按 token 在序列中的位置（0, 1, 2, ...）顺序遍历。给定一个 token 在序列中的位置 `i`：

```
i = 0:   逻辑部位在第 0 块,  块内偏移 0
i = 255: 逻辑部位在第 0 块,  块内偏移 255
i = 256: 逻辑部位在第 1 块,  块内偏移 0
i = 300: 逻辑部位在第 1 块,  块内偏移 44
i = 512: 逻辑部位在第 2 块,  块内偏移 0
```

即：

```
逻辑块号 = i // 256       (i ÷ 256 的整数部分)
块内偏移 = i % 256        (i ÷ 256 的余数)
```

然后通过查 `block_table` 把 逻辑块号 映射到 物理块号：

```
block_num = token_idx // 256
block_offset = token_idx % 256

物理块号 = block_tables[batch_idx, block_num]   ← 查映射表
K = k_cache[物理块号, block_offset, head, :]    ← 读数据
```

`//256` 和 `%256` 把一条长序列拆分成了块大小的片段，`block_table` 再把这些散落的片段串回连续顺序。这就是 PagedAttention 中 "paged" 的含义——和操作系统的虚拟内存分页是同一思想。

---

## Q8 · Decode 时 cache 中的 K、V 是分散在池子的不同位置，如何拼成完整的"K 矩阵"？

**答案是：它根本不拼。**

PagedAttention 采用**流式读取、即时计算**的方式，从不一次性把全部历史 K/V 加载到显存中组一个大矩阵。

以一条序列、一个 head 为例，context_len=300，block_size=256，block_table=[7, 3]：

```
Q = 当前新 token 的 query (head_dim=128, 1 个向量)

外层循环: chunk_idx 每轮处理 BLOCK_N=64 个历史 token

chunk_idx=0 (token 0~63):
    Token 0: block=0/256=0, off=0  → 物理块7, offset0  → K = k_cache[7, 0, head]  → Q·K
    Token 1: block=1/256=0, off=1  → 物理块7, offset1  → K = k_cache[7, 1, head]  → Q·K
    ...
    Token 63: block=63/256=0, off=63 → 物理块7, offset63 → K = k_cache[7, 63, head]
    对这 64 个 K 向量: 计算 Q·K^T → softmax → 累加 ×V

chunk_idx=1 (token 64~127):
    Token 64: block=64/256=0, off=64 → 物理块7, offset64 → K = k_cache[7, 64, head]
    ... (同上，全部落在物理块7)

chunk_idx=4 (token 256~299):
    Token 256: block=256/256=1, off=0 → 物理块3, offset0  → K = k_cache[3, 0, head]
    ...
    Token 299: block=1,  off=43 → 物理块3, offset43

最终: output = (累加 ×V) / softmax分母
```

每一轮只从 cache 中加载 64 个 K 向量和 64 个 V 向量到寄存器，算完就丢弃。**物理上零散的 block 通过 `block_table` 索引串成逻辑上的连续序列，从不创建完整的 `(300, 8, 128)` 的 K 矩阵。**

---

## Q9 · 如果同时有多条对话在进行，KV cache 如何区分属于哪条对话？

通过 `batch_idx`（批次索引）。每个 Sequence 维护自己的 `block_table`、`num_tokens` 等元数据。

在 prepare_decode 中（`model_runner.py:323-331`），所有序列的 block_table 拼成一个 `(batch_size, max_blocks)` 的二维张量：

```python
# 序列 A 的 block_table = [7, 3, -1]
# 序列 B 的 block_table = [5, 12, 8]

block_tables = torch.tensor([
    [7,  3, -1],   # batch_idx=0 → 序列 A
    [5, 12,  8],   # batch_idx=1 → 序列 B
])
```

在 paged attention kernel 中，每个 CUDA block 负责处理一个 `(batch_idx, head_idx)` 对：

```python
batch_idx = tl.program_id(0)
# 序列 A (batch_idx=0) 的 block_table[0]=[7,3,-1]
# 序列 B (batch_idx=1) 的 block_table[1]=[5,12,8]
```

序列 A 的 token_0 映射到物理块 7，序列 B 的 token_0 映射到物理块 5——两者物理存储隔离，互不干扰。唯一例外是 prefix cache 命中时，两个序列可能共享同一个物理块（通过 ref_count 引用计数管理）。

---

## Q10 · 为什么只缓存 K 和 V，不缓存 Q？

因为 Q 在后续步骤中**完全不再被需要**。

```
Token 128 生成时: 需要 Q₁₂₈、K₀~K₁₂₇、V₀~V₁₂₇
Token 129 生成时: 需要 Q₁₂₉、K₀~K₁₂₈、V₀~V₁₂₈
                   ↑ Q₁₂₈ 不再需要
```

不同 token 之间需要的 Q 完全独立，Q 只对"当前这个 token 的注意力计算"有意义，用完即可丢弃。而 K 和 V 的角色不同——历史 token 的 K 会被后续所有步骤的 Q 查询（作为"被查询的索引"），历史 token 的 V 会被后续所有步骤加权求和（作为"被查出的内容"）。

---

## Q11 · V 做缓存了吗？为什么叫 KV Cache？

**K 和 V 都做了缓存。** 从代码可以明确看出：

池子形状本身就包含了 K 和 V 两套：`(2, 28, ...)`，第 0 维的 `2` 代表 K 和 V。

写入时 (`attention.py:63-64`)：

```python
tl.store(k_cache_ptr + cache_offset, key)    # 写 K
tl.store(v_cache_ptr + cache_offset, value)  # 写 V
```

读取时，K 循环和 V 循环是分开的 (`attention.py:341-403`)：

```python
# 第一遍: 遍历全部 K，计算注意力权重 (softmax 分子/分母)
for i in range(BLOCK_N):
    K = k_cache[...]           # 从 cache 读 K
    score = Q · K^T            # 算注意力分数
    在线更新 softmax(score)     # 累加分子分母

# 第二遍: 遍历全部 V，用权重加权求和
for i in range(BLOCK_N):
    V = v_cache[...]           # 从 cache 读 V
    output += weight × V       # 加权累加
```

叫"KV Cache"是准确的——两者都被缓存，两者都在 decode 被读取。之所以 Q 不在名字里，是因为 Q 不需要缓存（见 Q10）。
