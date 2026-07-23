您指出的问题很对，第 1 节确实把“Node 是什么”和“为什么要有 Pod”搅在一起，显得臃肿且与后面章节的节奏不协调。下面按“先讲 Node 本身，再单独讲它与 Pod 的关系”的思路重新组织，同时把标题和行文收敛得更平实，避免拉得太长。

---

# 理解 Kubernetes Node

如果把 Kubernetes 比作一个系统，控制平面是大脑，Node 就是执行任务的四肢。Node 负责真正运行工作负载，将底层计算、存储、网络能力暴露给集群。

## 1. Node 是什么

**Node 是集群中的工作节点**，可以是物理机、虚拟机或云实例。每个 Node 上运行两个核心组件：

- **kubelet**：与控制平面通信，执行 Pod 的创建、启停等指令。
- **容器运行时**（如 containerd、CRI-O）：拉取镜像并运行容器。

Node 提供 CPU、内存、网络等物理资源，但它不直接管理零散的容器。Kubernetes 将紧密相关的容器封装成 Pod，再将 Pod 调度到 Node 上。因此，**Node 是 Pod 的载体**——一个 Node 可以运行多个 Pod，每个 Pod 内含一个或多个容器。

## 2. Node 与 Pod 的关系

之所以需要 Pod 这一层抽象，是因为现实中一个应用经常需要多个进程紧密协作。例如，Web 服务旁边往往跟着一个负责收集日志、更新证书的边车容器（sidecar）。这些辅助容器必须和主容器“同生共死”，并且最好能通过 localhost 通信、共享文件。如果把它们打进同一个 Pod，就能获得三个保证：

- 共享网络命名空间 → 共享同一个 IP，能用 localhost 互访，端口不能冲突 
- 共享存储卷（Volume） → 可以访问同一个目录交换文件 
- 始终被调度到同一个 Node → 不会出现一个在这台机器、一个在那台机器的情况

没有 Pod，就只能把所有进程塞进一个容器（违背单一职责），或手动用主机网络、自行确保它们落在同一节点，很麻烦。Pod 就是为此设计的“容器组”。

Pod 这种“逻辑主机”的抽象，很容易被拿来和 Docker Compose 的 service 比较，但它们的耦合程度不同。Compose 的 service 是独立的容器，各自拥有网络栈，不能通过 localhost 互访，依赖自定义网络通信；而 Pod 内所有容器共享网络和 IPC，就像同一台机器上的多个进程。打个比方：Pod 像是同一个人身上的器官，共享生命；Compose 的 service 则像同楼里不同房间的人，要通过楼道（网络）打招呼。

## 3. Node 对象的组成

用 `kubectl describe node` 查看节点，会看到这些关键部分：

### 3.1 状态

反映节点是否健康、能否接纳新 Pod。常见字段：

- **Conditions**：如 `Ready`、`MemoryPressure`、`DiskPressure`、`PIDPressure`、`NetworkUnavailable`。
- **Addresses**：节点 IP 和主机名。
- **Capacity 与 Allocatable**：Capacity 是节点总资源，Allocatable 是扣除系统保留后可用于 Pod 的部分。

### 3.2 资源容量

Node 会显式声明自己“有什么”：

```yaml
Capacity:
  cpu:    16
  memory: 64Gi
  nvidia.com/gpu: 4
```

这是调度决策的依据。

### 3.3 元数据

Node 也有 Labels（标签）和 Annotations（注解）。其中标签是连接节点能力与调度策略的桥梁。

## 4. 标签：为 Node 打上能力标识

**Labels** 是自由附加在对象上的键值对，用来描述节点的特殊属性：

- 地理位置：`topology.kubernetes.io/region: us-west`
- 硬件特性：`disktype: ssd`、`accelerator: nvidia-tesla-t4`
- 环境角色：`env: production`、`node-role.kubernetes.io/worker: true`
- 自定义特征：`gpu: on`、`team: ml-engineers`

标签不改变硬件本身，但向调度器提供语义化标记。即使节点插满 GPU，若没有打上 `gpu=on`，调度器也不会把它视为 GPU 节点。管理员可以动态修改标签，例如：

```bash
kubectl label nodes <node-id> gpu=on
```

这条命令不会改变节点的硬件，但会立即向控制平面声明：“该节点具备 GPU 能力，可以把需要 GPU 的 Pod 调度过来”。正是通过动态标签，集群的拓扑和能力信息才能实时跟上基础设施的真实状态。

## 5. 调度器如何利用标签

创建 Pod 时，调度器默认只看资源是否足够，但下面三种机制允许更精细地利用标签进行控制。

### 5.1 nodeSelector（最简单）

Pod spec 中指定：

```yaml
nodeSelector:
  gpu: "on"
```

Pod 只会被调度到带有 `gpu=on` 标签的节点，否则会一直 Pending。

### 5.2 Node Affinity（更丰富的表达）

支持硬性条件（required）和软性偏好（preferred）：

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: gpu
          operator: In
          values:
          - "on"
```

可以强制要求 `gpu=on`，也能表达“优先 SSD 节点”等倾向。

### 5.3 Taints & Tolerations（排斥与容忍）

如果说标签是“吸引”Pod，**Taint（污点）** 就是“驱离”Pod。给节点添加污点：

```bash
kubectl taint nodes <node-id> gpu=true:NoSchedule
```

这样只有显式声明了对应 **Toleration（容忍）** 的 Pod 才能调度上来。例如：

```yaml
tolerations:
- key: "gpu"
  operator: "Equal"
  value: "true"
  effect: "NoSchedule"
```

污点还支持 `PreferNoSchedule`（尽量不调度）和 `NoExecute`（驱逐已有 Pod）。实践中常将标签与污点配合：用标签标记“我有 GPU”，用污点阻止无关 Pod 抢占，保证 GPU 节点专用于特定负载。

---

Node 是资源基础，标签是把节点能力翻译给调度策略的桥梁。理解 Node，就是理解集群如何抽象、标记、选择和管理物理资源。许多操作看似简单，背后正是一整套这样优雅的设计在支撑。