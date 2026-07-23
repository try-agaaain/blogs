根据要求，文章已做如下修订：

1. **第1节拆分**：将“地基”一节细分为1.1～1.4四个小节，内容重新编排，消除重复，逻辑更连贯。
2. **序号统一**：所有章节序号改用数字（1、2、3……），原中文序号对应转换。
3. **流程图修正**：纠正了INPUT钩子内表顺序——按优先级应当是`mangle → nat → filter`，原流程图中`filter`误排在`nat`之前，现已修正。
4. **NAT概念补充**：在4.2节开头增加了NAT（网络地址转换）的本质定义，使其在讲解工作原理前先交代清楚是什么。

以下是修改后的全文。

---

# 深入理解 iptables：穿越 Netfilter 钩子的包过滤艺术

iptables 是 Linux 网络管理员最熟悉的工具之一，但若只停留在命令层面，往往难以应对复杂场景。本文将穿透表面，从内核 Netfilter 钩子与表/链的精确关系出发，重新绘制数据包在内核中的完整旅程，并逐层剖析连接追踪、NAT 实现、高级匹配以及性能调优，最终帮助你建立起一幅可运行于大脑的精确模型。

## 1. 地基：五个 Netfilter 钩子与五张表

iptables 是用户空间的配置工具，真正工作的是内核中的 **Netfilter** 框架。Netfilter 在网络协议栈中插入了五个**钩子（hook）**，每个钩子对应一个处理时机；同时用**表（table）**和**链（chain）**来组织规则，表代表功能类别，链则直接绑定到对应钩子。理解二者的精确关系是驾驭 iptables 的根基。

### 1.1 五个 Netfilter 钩子

数据包经过网络栈时，会依次碰到五个钩子：

1. **PREROUTING** — 数据包从网卡驱动进入，路由决策之前  
2. **LOCAL_IN** — 经过路由，判定发往本机，进入上层协议栈之前  
3. **FORWARD** — 经过路由，判定需要转发，离开本机之前  
4. **LOCAL_OUT** — 本机进程产生的数据包，路由决策之前  
5. **POSTROUTING** — 所有（本地产生或转发的）数据包离开网卡之前

### 1.2 五张表及其分工

当前内核支持五张表，其中核心的四张为：

- **raw** — 优先级最高，用于在连接追踪之前标记数据包，让指定流量跳过追踪（NOTRACK）。
- **mangle** — 用于修改数据包头（TOS、TTL、MARK 等），几乎可出现在所有钩子。
- **nat** — 实现网络地址转换（SNAT/DNAT/MASQUERADE 等），仅在需要地址转换的路径上存在。
- **filter** — 负责数据包过滤（ACCEPT/DROP/REJECT），这是大多数场景的主战场。
- **security** — 用于 SELinux 等强制访问控制，使用较少，本文略过。

**为什么需要多张表，而不是用一张 mangle 包揽一切？**  
mangle 表确实可以挂在几乎所有钩子上，也能修改数据包的任意字段，但内核网络栈讲究“正确的时机做正确的事”，并要求清晰的逻辑分离。五张表的根本区别在于它们的**优先级**和**意图**：

- **raw** 表优先级最高，在连接跟踪启动之前运行，专门用来标记“不必跟踪”的包，决定是否进入跟踪系统；
- **mangle** 次之，负责纯粹的包内容修改，不影响连接状态和路由决策的“含义”；
- **nat** 表承载地址转换，其中的 DNAT 会改变目标地址，直接影响后续路由决策，因此必须恰好在路由之前完成；
- **filter** 表优先级最低，只负责最终的“放行或丢弃”判断，不修改包内容。

这种分离让每张表职责单一、规则顺序可控，也便于内核在不需要某类功能时完全跳过对应表，提高效率。若所有功能混在同一优先级内，规则顺序将难以管理，也无法构建“先决定是否跟踪→再修改→再转换地址→最后过滤”这种刚性的处理流水线。

### 1.3 表与链的绑定逻辑

表之所以只出现在特定钩子上，完全由其功能在数据包旅程中的位置决定：

- **raw 表只绑定 PREROUTING 和 OUTPUT**：连接跟踪的入口点恰好在这两个钩子之后，raw 要赶在跟踪之前标记 NOTRACK，因此只需也仅能在这两处存在。
- **nat 表绑定 PREROUTING、OUTPUT、POSTROUTING（及极少使用的 INPUT）**：DNAT（改目标地址）必须在路由决策前完成，所以位于 PREROUTING 和 OUTPUT；SNAT/MASQUERADE（改源地址）必须在数据包离开主机的最后一步完成，所以位于 POSTROUTING。FORWARD 钩子本身不涉及地址转换，因此 nat 表没有 FORWARD 链。
- **filter 表只绑定 INPUT、FORWARD、OUTPUT**：过滤意在控制“是否允许通过”，判断点在路由之后——本地接收（INPUT）、转发（FORWARD）或本机发出（OUTPUT）。PREROUTING 和 POSTROUTING 阶段尚未确定包的最终去向，不适合做终极访问控制。
- **mangle 表覆盖几乎所有钩子**：包的修改需求可能出现在任何环节，因此几乎每处都可使用，但它不参与“是否跟踪”和“是否允许”的终极决策。

这些关系总结为下表（务必记牢）：

| 链（钩子）  | raw  | mangle | nat  | filter |
| ----------- | ---- | ------ | ---- | ------ |
| PREROUTING  | ✓    | ✓      | ✓    | ✗      |
| INPUT       | ✗    | ✓      | (✓)¹ | ✓      |
| FORWARD     | ✗    | ✓      | ✗    | ✓      |
| OUTPUT      | ✓    | ✓      | ✓    | ✓      |
| POSTROUTING | ✗    | ✓      | ✓    | ✗      |

¹ `nat` 表的 `INPUT` 链极少使用，仅在特定场景下对本机接收的包做 DNAT。

从表中可以立即看出：**filter 没有 PREROUTING 和 POSTROUTING 链**，因此不能在路由前过滤转发包；**nat 没有 FORWARD 链**，转发包的地址转换必须安排在 PREROUTING（DNAT）或 POSTROUTING（SNAT）。

### 1.4 钩子内表的优先级顺序

同一个钩子上注册的多张表有严格的执行顺序，由 Netfilter 的优先级决定，从早到晚依次为：**raw → mangle → nat → filter**。例如，当数据包到达 PREROUTING 钩子时，内核先执行 raw 表的 PREROUTING 链，然后是 mangle 表，最后是 nat 表；filter 表根本没有在 PREROUTING 注册，因此绝无可能在此处执行。这一顺序往往是排查“为何我的规则没生效”的关键线索。

## 2. 数据包的完整旅程

综合上述钩子与表的关系，可以得到一幅精确无歧义的数据包流向图。同一钩子内，表按优先级从上到下排列：

```
                 入站包
                    │
                网卡接收
                    │
        PREROUTING(raw)
                    │
             Conntrack
                    │
       PREROUTING(mangle→nat)
                    │
              路由决策
              /       \
             /         \
            /           \
       本机接收       转发包
           │             │
INPUT(mangle→filter)  FORWARD(mangle→filter)
           │             │
        本地进程          │
                         │
                  POSTROUTING(mangle→nat)
                         │
                      网卡发送 
本地进程
    │
OUTPUT(raw)
    │
Conntrack
    │
OUTPUT(mangle→nat→filter)
    │
路由决策
    │
POSTROUTING(mangle→nat)
    │
网卡发送

```

## 3. 规则精要：匹配与目标

一条 iptables 命令的骨架为：

```
iptables -t <表> -A <链> [匹配条件...] -j <动作>
```

表默认是 `filter`，`-A` 表示追加。规则沿着链从上到下顺序匹配，命中后执行动作（`-j`），若未命中则继续下一条，最终抵达链的默认策略（仅 filter 表的链有自定义默认策略，其他表固定为 ACCEPT 或 RETURN）。

### 3.1 匹配条件——为何需要，以及如何使用

如果说链决定“在哪个时机处理”，动作决定“如何处理”，那么匹配条件就是“处理谁”的唯一凭证。没有匹配条件，规则将作用于所有通过的数据包，这几乎永远会出错。匹配条件的核心任务是为数据包画像，从粗到精层层限定：

- **基本条件**——接口（`-i`、`-o`）、源/目标 IP（`-s`、`-d`）、协议（`-p`）和端口（`--dport`、`--sport`）——构成日常过滤的骨架，直白高效。
- **扩展匹配模块**——如 `multiport`、`ipset`、`conntrack`、`recent` 等——用于解决基本条件难以表达或效率不足的问题。例如，用 `multiport` 一条规则匹配多个离散端口；用 `ipset` 的哈希查找高效匹配数千个 IP 段；而 `conntrack` 的状态匹配更是有状态防火墙的灵魂，让“允许所有回包”只用一条规则。

下列常用模块从不同维度扩展了数据包描述能力：

- **状态匹配**（`-m conntrack --ctstate`）：有状态防火墙的基石，详述于下一节。
- **multiport**：`-m multiport --dports 22,80,443`，一条规则抵多条。
- **ipset**：`-m set --match-set blacklist src`，可高效匹配海量 IP 或网段，远胜线性规则。
- **limit / hashlimit**：`-m limit --limit 5/min` 用于限速日志或抑制暴力；`hashlimit` 可按源 IP 等分桶独立限速。
- **recent**：动态列表，常用于防止 SSH 爆破，结合 `--hitcount` 和 `--seconds`。
- **string / u32**：应用层或深层头部匹配，消耗 CPU 较高，慎用。
- **owner**：`-m owner --uid-owner 1000`，只能用于 OUTPUT 链，限定本地进程属主。
- **time**：按时间段生效，如 `--timestart 09:00 --timestop 18:00`。

### 3.2 动作目标——决定数据包的命运

常见目标：

- **ACCEPT** / **DROP** / **REJECT**：通过、静默丢弃、明确拒绝（返回 ICMP 差错）。
- **LOG** / **NFLOG**：内核日志或用户态队列日志，调试利器。
- **SNAT** / **MASQUERADE** / **DNAT** / **REDIRECT**：地址转换家族，必须用在 `nat` 表。
- **MARK**：打标签（`--set-mark`），用于策略路由，只能用在 `mangle` 表。
- **NOTRACK**：仅在 `raw` 表，跳过连接追踪。
- **TRACE**：在 `raw` 表激活包跟踪，内核日志会打印数据包遍历的每一张表链，排查的终极手段。

## 4. 连接追踪：状态防火墙与 NAT 的幕后引擎

iptables 能够实现“有状态”过滤，全赖内核连接跟踪系统（`nf_conntrack`）。它对每个经过的数据包分配连接状态，并维护一张连接跟踪表。这张表既是 NAT 翻译的根基，也是状态匹配的数据来源。

### 4.1 状态定义

- **NEW** — 连接的第一个有效包（如 TCP SYN，或 UDP 首个数据包）。
- **ESTABLISHED** — 已经收到反向应答，连接建立完成。
- **RELATED** — 与已存在连接关联的衍生连接，例如 FTP 数据通道、ICMP 差错报文（需加载对应辅助模块如 `nf_conntrack_ftp`）。
- **INVALID** — 状态异常或无法归类，可能来自端口扫描或乱序数据包。
- **UNTRACKED** — 在 `raw` 表用 `NOTRACK` 显式跳过跟踪的流量。

一条经典的放行规则：

```
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
```

它允许所有已建立连接的回包进入，配合默认策略 DROP，可极大精简规则集。注意必须同时允许 `RELATED`，否则 ICMP “主机不可达”等关键差错会被丢弃，FTP 等复杂协议也会失败。

### 4.2 NAT 如何工作

**NAT（Network Address Translation，网络地址转换）** 是一种修改数据包源或目标 IP 地址（以及端口）的技术，核心目的是实现内网私有地址与公网地址之间的复用或隐藏，也可用于负载均衡、透明代理等场景。Netfilter 的 NAT 实现完全依赖连接跟踪：内核在首次数据包通过时创建地址映射并写入跟踪表，后续双向数据包均可根据该映射自动完成地址转换。

- **SNAT/MASQUERADE**（源地址转换）工作在 POSTROUTING 钩子的 `nat` 表。内核将数据包源地址改为指定公网 IP（或动态接口地址），并在跟踪表中记录“内网 IP:端口 ↔ 公网 IP:端口”的映射。返回包在 PREROUTING 钩子自动进行逆转换，无需额外规则。
- **DNAT**（目标地址转换）工作在 PREROUTING 或 OUTPUT 钩子的 `nat` 表。因为路由决策基于最终目标地址进行，所以 DNAT 必须在路由之前完成。例如端口映射规则：

```
iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to-destination 192.168.1.10:8080
```

- **REDIRECT** 是本机端口重定向，透明代理常使用，实质是特殊的 DNAT，同样依赖 `nat` 表和连接跟踪。

### 4.3 连接跟踪的性能影响

高并发场景下，连接跟踪表可能成为瓶颈。可通过 `/proc/sys/net/netfilter/nf_conntrack_max` 调整最大表项数。若不需要状态跟踪（如 DNS 服务器仅处理无连接 UDP），可在 `raw` 表设置 NOTRACK，并在 filter 中用 `UNTRACKED` 状态放行，避免 CPU 开销和表项占用。

## 5. 核心实战场景

### 5.1 基础有状态防火墙

```bash
# 默认策略
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT
# 允许回包
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
# 开放服务端口
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -m conntrack --ctstate NEW -j ACCEPT
# 允许本地回环
iptables -A INPUT -i lo -j ACCEPT
```

### 5.2 NAT 网关

```bash
# 启用转发
sysctl -w net.ipv4.ip_forward=1
# SNAT（静态公网 IP）
iptables -t nat -A POSTROUTING -s 192.168.1.0/24 -o eth0 -j SNAT --to-source 203.0.113.5
# 或 MASQUERADE（动态 IP）
iptables -t nat -A POSTROUTING -s 192.168.1.0/24 -o eth0 -j MASQUERADE
# 转发过滤
iptables -A FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A FORWARD -s 192.168.1.0/24 -p tcp --dport 80 -j ACCEPT
```

### 5.3 防暴力破解

```bash
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --set --name SSH
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent \
    --update --seconds 60 --hitcount 4 --name SSH -j DROP
```

### 5.4 基于策略路由的双线负载

```bash
# 在内网口打标记
iptables -t mangle -A PREROUTING -i eth1 -s 10.0.1.0/24 -j MARK --set-mark 1
# 添加路由表
ip rule add fwmark 1 table line1
ip route add default via 10.0.1.1 table line1
```

### 5.5 透明代理（REDIRECT）

```bash
iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 80 -j REDIRECT --to-port 3128
```

所有进入的 80 端口流量被重定向到本机 3128 端口代理程序，客户端无感知。

## 6. 调试与性能优化

### 6.1 包路径跟踪

**TRACE** 功能是透视规则处理的终极武器：

```bash
modprobe nf_log_ipv4
iptables -t raw -A PREROUTING -p tcp --dport 22 -j TRACE
iptables -t raw -A OUTPUT -p tcp --sport 22 -j TRACE
```

`dmesg` 或日志中会打印出包依次经过的每一个钩子、每一张表和链，以及是否命中某条规则，能瞬间澄清“规则为什么没匹配”的疑问。

**LOG** 目标是临时插入的简易探针，可在怀疑的链上分段插入：

```bash
iptables -A INPUT -s 192.168.1.100 -j LOG --log-prefix "INPUT from .100: "
```

观察 `/var/log/kern.log` 即可确认数据包是否到达该链。

### 6.2 计数器分析

`iptables -L -n -v` 列出每条规则匹配的包数和字节数。连续运行并观察计数器变化，可判断流量是否被预期规则命中，还是落入了默认策略。

### 6.3 连接跟踪表查询

```bash
conntrack -L    # 列出所有跟踪连接
conntrack -E    # 事件模式，实时显示新建/销毁
```

可验证 NAT 映射是否正确，或诊断表项溢出。

### 6.4 性能原则

- **善用 ipset**：上万条 IP 规则用 ipset 代替线性匹配，性能提升显著。
- **最常匹配的规则置于顶部**：`ESTABLISHED` 放行规则应为 INPUT 链第一条。
- **不必要的连接跟踪用 raw 表跳过**：如高流量 UDP 服务。
- **避免频繁规则变更**：推荐使用 `iptables-restore` 一次性原子加载完整规则集，既安全又快速。
- **监控连接表大小**：`nf_conntrack_count` 是关键指标，接近最大值时需扩容或排查异常连接。

---

现代系统中，`iptables` 命令底层可能已通过 iptables-nft 兼容层运行，Docker、Kubernetes 等平台也仍大量操作 iptables 规则。因此，即便你逐渐转向 nftables，透彻理解本文中的钩子模型、状态跟踪与 NAT 流程，仍是排查容器网络、虚拟化和传统服务器防火墙问题的必备能力。