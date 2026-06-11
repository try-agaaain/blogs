# 数据的生命线：Kubernetes PV 与 PVC 深度解析

当你创建一个 Pod，它的文件系统随容器而生，也随容器而灭——镜像只读层之上的可写层不过是临时的速写本。但对于数据库、消息队列、文件存储等有状态负载来说，数据不能丢、不能乱、不能停。Kubernetes 以 PV（PersistentVolume）和 PVC（PersistentVolumeClaim）两把钥匙，打开了容器世界通往持久化存储的大门。

本文聚焦 PV 与 PVC 这一对共生概念，深入它们的规范细节、绑定机制、生命周期的每一个阶段，以及真实的踩坑与最佳实践。

---

## 1. 为什么需要 PV 和 PVC

容器的临时性是其设计的核心假设：Pod 可以随时销毁和重建，节点可以随时被下线，但数据必须活得比 Pod 长。

理解 PV/PVC 之前，先看清楚它们要解决什么：

| 问题 | 表现 |
|------|------|
| **存储耦合** | 直接在 Pod 里写死存储路径和后端信息（如 NFS 地址），更换存储需逐个修改 Pod |
| **缺乏抽象** | 开发者必须知道底层存储是什么、在哪里，而不是只声明"我需要 10Gi 快速存储" |
| **生命周期不一致** | Pod 删除后存储可能被一起清理，或变成无人管理的孤儿资源 |

PV 和 PVC 的引入，把存储分割为两个清晰的职责域：

- **管理员**（集群运维者）：准备存储资源，定义容量、类型、回收策略。
- **开发者**（应用作者）：声明存储需求，不关心底层的实现细节。

两者通过一个声明式的匹配机制联系起来——这是 Kubernetes 一贯的设计哲学：**用声明替代沟通，用契约替代耦合**。

---

## 2. PV：集群中的存储资源

PersistentVolume 是集群级别的资源，独立于任何 Pod 和节点存在。它代表了一块真实的存储——可能是 NFS 共享目录、Ceph RBD 块设备、GCE Persistent Disk，或者是开发机上的一个 `hostPath` 目录。

### 2.1 PV 的定义

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: example-pv
spec:
  capacity:
    storage: 50Gi
  volumeMode: Filesystem
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: slow
  mountOptions:
    - hard
    - nfsvers=4.1
  nfs:
    server: nfs-server.internal
    path: /exports/data
```

每个 PV 的 `spec` 定义了一组关键属性，它们决定了 PV "是什么"以及"能被谁使用"。

### 2.2 容量

```yaml
capacity:
  storage: 50Gi
```

PV 声明的容量是硬约束——Kubernetes 不会超额分配。PVC 只要申请小于等于这个值的存储即可绑定，但无法请求超过 PV 容量的空间（包括未来扩容操作也被这块天花板限制）。

### 2.3 访问模式（Access Modes）

访问模式描述的是存储卷可以被多少节点以什么方式挂载。它依赖于底层存储后端的物理能力：

| 模式 | 缩写 | 含义 | 典型支持的后端 |
|------|------|------|----------------|
| `ReadWriteOnce` | RWO | 单节点读写 | AWS EBS、GCE PD、Azure Disk、hostPath |
| `ReadOnlyMany` | ROX | 多节点只读 | NFS、CephFS、GCE PersistentDisk |
| `ReadWriteMany` | RWX | 多节点读写 | NFS、CephFS、GlusterFS、Longhorn |
| `ReadWriteOncePod` | RWOP | 单 Pod 读写（v1.22+） | CSI 驱动（需支持） |

**RWO** 是最常见的模式，但有个微妙的限制：它允许**一个节点**上的**多个 Pod** 同时读写同一个卷（只要这些 Pod 都在同一节点上）。如果需要严格限制"单个 Pod"独占，请使用 `ReadWriteOncePod`。

**RWX** 最灵活，但也最危险——多个 Pod 同时写入同一文件系统可能导致数据损坏，除非应用自身实现了文件锁（如 NFS 的 `flock`）或使用数据库层面的并发控制。

**访问模式是在 PV 创建时固定的，PVC 只能请求一个子集**。比如 PVC 声明 `[ReadWriteOnce]` 可以绑定一个支持 `[ReadWriteOnce, ReadOnlyMany]` 的 PV，反之不行。

### 2.4 回收策略（Reclaim Policy）

当 PVC 被删除后，与之绑定的 PV 应该如何处理？这就是 `persistentVolumeReclaimPolicy` 的管辖范围：

- **Retain**（保留）：PV 进入 `Released` 状态，数据完整保留。但该 PV 不能再被其他 PVC 绑定，需要管理员手动清理和重新使用。管理员需依次执行：删除 PV → 清理后端存储数据 → 重新创建 PV。

- **Delete**（删除）：PV 和后端存储会被自动删除。适用于云提供商的动态卷（AWS EBS、GCE PD 等）。这是 StorageClass 动态供给时的默认策略。

- **Recycle**（已废弃）：曾经会执行 `rm -rf /volume/*` 并让 PV 回到 `Available` 状态，但从 v1.9 开始已被标记为废弃，由动态供给替代。

选择策略的核心问题是：**数据删掉后还能找回来吗？** 生产环境通常结合备份策略使用 `Retain`，或是用 `Delete` 配合快照实现回滚能力。


### 2.5 存储类别（StorageClass）

```yaml
storageClassName: slow
```

当 PV 设置了 `storageClassName`，它就只能与声明了相同类名的 PVC 绑定。一个没有 `storageClassName` 的 PV（空字符串）只能与也不指定类名的 PVC 匹配——但这在不同版本中行为有微妙差异，后面会细说。

### 2.6 卷模式（Volume Mode）

v1.13 引入的 `volumeMode` 区分了两种使用方式：

- **Filesystem**（默认）：卷被格式化为文件系统并挂载到目录。Pod 中通过 `mountPath` 访问。
- **Block**：卷以原始块设备的形式呈现，Pod 中使用 `volumeDevices` 将其映射为设备文件（如 `/dev/sdb`）。适合数据库等需要直接管理存储布局的应用。

```yaml
# 块卷的 PV 定义
apiVersion: v1
kind: PersistentVolume
metadata:
  name: block-pv
spec:
  capacity:
    storage: 100Gi
  volumeMode: Block
  accessModes:
    - ReadWriteOnce
  local:
    path: /dev/sdb
```

```yaml
# Pod 中使用块卷
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: mysql
    volumeDevices:
    - devicePath: /dev/sdb
      name: data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: block-pvc
```

块卷的好处是零文件系统开销、应用可完全控制数据布局，代价是不能直接用 `cp` 或 `ls` 操作。

### 2.7 挂载选项（Mount Options）

```yaml
mountOptions:
  - hard
  - nfsvers=4.1
```

只有 PV 支持手动指定挂载选项，PVC 和 Pod 无法覆盖。对于 NFS 卷，合理的挂载选项直接影响可用性和性能——例如 `hard,intr` 确保 NFS 服务器宕机时应用进程挂起而非静默写丢数据。

---

## 3. PVC：用户对存储的声明

PersistentVolumeClaim 是用户在命名空间级别发出的存储请求。它不关心存储在哪台机器、是什么品牌——它只描述需求。

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: example-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: slow
  volumeMode: Filesystem
```

### 3.1 PVC 的关键字段

| 字段 | 说明 | 匹配逻辑 |
|------|------|----------|
| `accessModes` | 需要的访问模式 | PV 必须包含 PVC 请求的全部模式 |
| `resources.requests.storage` | 所需容量 | PV 容量 ≥ PVC 请求容量（不检查精确匹配） |
| `storageClassName` | 存储类别 | PV 的 `storageClassName` 必须完全一致 |
| `selector` | 标签选择器 | 可选，进一步过滤 PV，支持 `matchLabels` 和 `matchExpressions` |
| `volumeMode` | Filesystem 或 Block | 必须与 PV 一致 |

### 3.2 Selector：精准选择

当多个 PV 都匹配容量和访问模式时，可以用 `selector` 做精细化筛选：

```yaml
spec:
  selector:
    matchLabels:
      storage-tier: ssd
      environment: production
```

PV 必须拥有所有指定的标签才能匹配。这相当于给存储资源打上业务维度的标签——"这个 PV 是给生产环境 SSD 类型的业务用的"。

### 3.3 PVC 的状态机

```
Pending  →  Bound  →  (被 Pod 使用)  →  (PV 回收后) Released / Lost
```

- **Pending**：PVC 已创建，但尚未找到匹配的 PV。可能原因：没有合适的 PV、StorageClass 动态供给尚未完成、或 `volumeBindingMode: WaitForFirstConsumer` 在等待 Pod 调度。
- **Bound**：已成功绑定到某个 PV。此时 PV 的状态也变为 `Bound`。
- **Released**：PVC 被删除，PV 进入 `Released`（仅当回收策略为 `Retain`）。此时 PV 还不能被复用，管理员需手动处理。
- **Lost**：PV 指向的后端存储不可达或已不存在。这是一个需要人工介入的错误状态。

---

## 4. 绑定机制：插座与插头的精确匹配

PV 和 PVC 的绑定是 Kubernetes 控制平面中 PersistentVolume Controller 的核心工作。这个控制循环的逻辑并不复杂，但细节藏在匹配条件里。

### 4.1 匹配条件

一个 PVC 要绑定一个 PV，必须同时满足以下条件：

```
PVC.accessModes  ⊆  PV.accessModes       （1）
PVC.storage      ≤  PV.capacity.storage  （2）
PVC.storageClassName == PV.storageClassName （3）
PVC.volumeMode   == PV.volumeMode        （4）
PV 的 label 覆盖 PVC.selector 的要求    （5）
PV 的状态为 Available                     （6）
```

条件（1）是子集关系：PV 可以支持更多访问模式，PVC 只要自己需要的那些被涵盖即可。例如 PV 支持 `[RWO, ROX]`，PVC 请求 `[ROX]` 即可绑定。

条件（2）是大小比较：PV 可以比 PVC 大，但不能小。这是"够用即可"的匹配哲学。

条件（3）的匹配规则有一个关键细节：**未设置 `storageClassName`（空字符串）与未指定 `storageClassName`（字段不存在）是不同的！**

| PVC.storageClassName | PV.storageClassName | 是否匹配 |
|----------------------|---------------------|----------|
| `"standard"` | `"standard"` | ✅ |
| `"standard"` | `"fast"` | ❌ |
| `""`（空字符串） | `""`（空字符串） | ✅ |
| 未设置 | 未设置 | ✅（仅在启用 DefaultStorageClass 准入插件时，会被赋予默认 StorageClass） |
| `""`（空字符串） | 未设置 | ❌ |

用一句话记忆：**PVC 没写 storageClassName 不等于空字符串，它会被默认 StorageClass 自动填充；如果想明确使用无类别的 PV，需显式写成 `storageClassName: ""`。**

### 4.2 绑定过程

1. PVC 被创建，PersistentVolumeController 收到通知。
2. 控制器遍历所有状态为 `Available` 的 PV。
3. 逐一检查上述五个匹配条件。
4. 找到第一个完全匹配的 PV 后，将 PV 的状态改为 `Bound`，并在 PV 的 `spec.claimRef` 中记录 PVC 的引用信息（命名空间 + 名称）。
5. PVC 的状态更新为 `Bound`，`spec.volumeName` 字段填入 PV 名称。

绑定是**排他的一对一关系**：一个 PV 同一时间只能绑定一个 PVC。一个 PVC 也只能绑定一个 PV。这是 Kubernetes 有意为之的简化——不需要考虑共享卷的多路绑定问题，那交给访问模式和存储后端去处理。

### 4.3 容量匹配的误区

很多人以为 PVC 申请 20Gi 会精确匹配一个 20Gi 的 PV，但 Kubernetes 实际的行为是：**找到容量 >= 20Gi 的最小可用 PV**。如果有 10Gi、20Gi、50Gi 三个 PV 都可用，它会优先匹配 20Gi 的那个（最接近但不小于请求值）。但这取决于遍历顺序，Kubernetes 并不保证"最小最优"——如果 50Gi 的 PV 先被遍历到，它也可能被绑定。

这个"最先匹配而非最优匹配"的行为，在静态供给场景下可能导致大 PV 被小 PVC 占用。解决方式：

- 使用 StorageClass 动态供给，按需创建精确大小的 PV。
- 借助 `selector` 和标签精确控制匹配范围。

---

## 5. PV 和 PVC 的完整生命周期

### 5.1 供给（Provisioning）

**静态供给**：管理员预先准备一批 PV。适合存储后端可控、使用量可预测的场景。

```bash
kubectl apply -f pv.yaml    # 创建 PV
kubectl apply -f pvc.yaml   # 创建 PVC → 自动绑定
```

**动态供给**：PVC 指定 `storageClassName`，系统自动调用 provisioner 创建 PV。无需管理员手动介入。

```yaml
# 只需声明 PVC，PV 会自动创建
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dynamic-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast    # 触发动态供给
```

动态供给是现代 Kubernetes 集群的标配——云上的 EBS、Azure Disk、GCP PD 都通过内置 provisioner 实现自动创建。但有个容易被忽略的细节：**动态创建出来的 PV 的回收策略默认继承自 StorageClass 的定义**，通常是 `Delete`。这意味着删除 PVC 时，云硬盘也会被删掉——如果只是想临时释放 PVC，务必先确认数据已备份。

### 5.2 绑定（Binding）

绑定瞬间完成。PV 从 `Available` → `Bound`，PVC 从 `Pending` → `Bound`。此时 PV 的 `claimRef` 会锁定到特定 PVC，防止其他 PVC 意外绑定。

如果想解绑或强制绑定，不要直接修改 `claimRef`——那是在绕过 Kubernetes 的内部状态机，极可能导致资源泄露。正确的做法是先删除 PVC，再清理 PV 的 `claimRef`，然后重新创建 PVC。

### 5.3 使用（Using）

Pod 通过 `volumes.persistentVolumeClaim` 引用 PVC：

```yaml
kind: Pod
spec:
  containers:
  - name: app
    volumeMounts:
    - name: data
      mountPath: /var/lib/data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: example-pvc
```

Pod 调度到节点后，kubelet 负责执行实际的挂载操作：

1. 如果卷尚未挂载到本节点，调用 CSI 或 in-tree 插件执行 `NodeStageVolume`（格式化、挂载到全局目录）。
2. 执行 `NodePublishVolume`，将卷绑定到 Pod 的挂载命名空间。
3. 容器启动后，在 `mountPath` 可访问存储。

如果在 Pod 运行期间删除 PVC（不推荐），Kubernetes 的最终一致性保证会导致：Pod 继续使用已挂载的卷，但无法再调度到新节点。因此**删除 PVC 前务必确保所有使用它的 Pod 已被删除或不再需要该存储**。

### 5.4 释放（Releasing）

删除 PVC 后，PV 进入释放阶段。回收策略开始生效：

| 策略 | 行为 | PV 最终状态 | 数据 |
|------|------|-------------|------|
| Retain | 保留 PV 和后端存储 | `Released` | 保留 |
| Delete | 删除 PV 和后端存储 | 消失 | 删除 |
| Recycle（废弃） | 清空数据后回到池中 | `Available` | 清空 |

**Retain** 的 PV 在 `Released` 状态是一个"灰色地带"：数据尚在，但该 PV 不能被重新绑定。管理员必须手动执行以下步骤才能复用：

```bash
kubectl delete pv <pv-name>          # 删除 PV 对象
# 验证后端存储数据是否完整（NFS/Ceph/云硬盘）
kubectl apply -f pv-recreate.yaml    # 重新创建 PV（无 claimRef）
```

这个过程极易出错——很多人忘记检查后端数据就重建 PV，导致数据覆盖。

### 5.5 扩展（Volume Expansion）

如果 PVC 声明的容量不够用了，Kubernetes 支持在线扩展（需 `StorageClass.allowVolumeExpansion: true`）：

```yaml
# StorageClass 中开启
allowVolumeExpansion: true
```

```bash
# 直接修改 PVC 的 storage 值
kubectl edit pvc example-pvc
# 修改 spec.resources.requests.storage: 20Gi → 40Gi
```

扩展过程对于大多数块存储（EBS、Cinder 等）是**在线完成**的——Pod 不需要重启。但对于文件系统，kubelet 会自动调用 `resizeFs` 来扩展现有文件系统以匹配新容量。扩展当前不支持缩容。

---

## 6. 动态供给与 StorageClass 的协作

PV/PVC 的静态供给就像去超市买米——米已经摆在货架上（PV 已创建），你只需拿一袋合适的下来（创建 PVC 匹配）。动态供给则是现碾现卖：你告诉店员要多少米（创建 PVC），店员去仓库碾出来（provisioner 创建 PV）。

StorageClass 是连接 PVC 和动态供给的桥梁：

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
reclaimPolicy: Delete
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

关键字段 `volumeBindingMode` 值得展开：

- **`Immediate`**：PVC 创建后立即尝试创建 PV 并绑定。在跨可用区的云环境中，这可能导致 PV 创建在一个可用区，而 Pod 被调度到另一个——产生跨 AZ 的 I/O 延迟和流量费用。
- **`WaitForFirstConsumer`**：延迟到 Pod 被调度后再创建 PV。绑定过程变为：PVC 创建 → Pending → Pod 被调度 → 调度器在选定的节点上创建 PV → PVC/PV 绑定。这确保了 PV 与 Pod 在同一个可用区。

```yaml
# 使用 WaitForFirstConsumer 的典型效果
# 1. 创建 PVC → 状态为 Pending（不立即绑定）
# 2. 创建 Pod 并引用该 PVC
# 3. 调度器选出目标节点（含可用区信息）
# 4. 只在目标可用区创建 PV 并绑定
# 5. Pod 启动，挂载 PV
```

这个模式对云上数据库类工作负载尤其重要——跨 AZ 的 EBS 卷无法挂载，如果没有 `WaitForFirstConsumer`，Pod 会一直 Pending 且错误信息令人困惑。

---

## 7. 本地 PV：当性能优先

云存储提供了便利性和跨节点挂载能力，但网络存储的延迟始终高于本地 SSD。对于追求极致 I/O 的场景（如 etcd、Elasticsearch 热节点），Kubernetes 提供了 **Local PersistentVolume**：

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: local-pv
spec:
  capacity:
    storage: 100Gi
  volumeMode: Filesystem
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: local-ssd
  local:
    path: /mnt/disks/ssd1
  nodeAffinity:
    required:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - node-01
```

本地 PV 使用 `nodeAffinity` 绑定到特定节点，因为只有该节点上有对应的本地磁盘。这带来一个重要的行为差异：

- **Pod 调度必须考虑节点亲和性**：使用本地 PV 的 Pod 会被隐式约束到 PV 所在的节点。如果节点故障，Pod 无法在其他节点上恢复——因为本地磁盘拔不走。
- **回收策略建议 `Retain`**：`Delete` 不会删除本地磁盘（因为它不是云硬盘），但 PV 会被移除，导致数据不可访问。

本地 PV 的高性能来自**牺牲了存储与节点的解耦**——这正是权衡的体现。

---

## 8. 常见陷阱与实践建议

### 陷阱一：跨 AZ 绑定导致 Pod 一直 Pending

**现象**：PVC 已 Bound，但 Pod 一直 Pending，`kubectl describe pod` 提示卷挂载失败。

**原因**：`volumeBindingMode: Immediate` 导致 PV 在 AZ-A 创建，但 Pod 被调度到 AZ-B。云提供商的块存储（如 EBS）不能跨 AZ 挂载。

**解法**：使用 `WaitForFirstConsumer` 或手动配置 `allowedTopologies`。

### 陷阱二：PVC 删除后 PV 被意外删除

**现象**：本想临时释放 PVC，结果 PV 和后端存储一起消失了。

**原因**：回收策略为 `Delete`，且 StorageClass 动态供给的默认回收策略就是 `Delete`。

**解法**：对重要数据，在使用动态供给时，可以先创建 PVC 获得 PV，然后修改 PV 的回收策略为 `Retain`：

```bash
kubectl patch pv <pv-name> -p '{"spec":{"persistentVolumeReclaimPolicy":"Retain"}}'
```

### 陷阱三：静态 PV 的容量碎片化

**现象**：创建了 100Gi 的 PV，被只需要 1Gi 的 PVC 绑定，导致大容量 PV 被低效使用。

**原因**：PV 匹配是"先到先得"，没有容量排序或预留机制。

**解法**：使用动态供给或使用多个 small PV 配合 selector 分类。

### 陷阱四：NFS 权限问题

**现象**：Pod 以非 root 用户运行，但在 NFS 挂载目录下没有写入权限。

**原因**：NFS 的权限映射基于 UID/GID，容器内用户的 UID 可能没有 NFS 共享目录的写入权限。

**解法**：在 PV 中设置 `mountOptions` 里的 `uid`/`gid`，或在 NFS 服务端调整导出权限。

### 实践建议总结

| 场景 | 推荐做法 |
|------|----------|
| 开发/测试 | 动态供给 Default StorageClass，`Delete` 回收 |
| 生产数据库 | `WaitForFirstConsumer` + `Retain` 回收 + 定期快照 |
| 共享文件 | NFS/CephFS，RWX 模式，注意并发写入风险 |
| 高性能本地存储 | Local PV + `Retain` + 节点亲和性 |
| 跨可用区高可用 | 使用支持跨 AZ 复制的存储（如云厂商的 NAS 或复制卷） |

---

## 9. 全景回顾

PV 和 PVC 的设计，本质上是对"谁来准备存储、谁来使用存储"这两个问题的清晰划分：

```
┌─────────────────────────────────────────────────┐
│              集群管理员职责                        │
│                                                   │
│  创建 PV（静态）或定义 StorageClass（动态）          │
│  决定：容量、访问模式、回收策略、后端类型              │
└──────────────────────┬──────────────────────────┘
                       │  PV 作为集群资源存在
                       ▼
┌─────────────────────────────────────────────────┐
│             PersistentVolume Controller          │
│    匹配 PV ↔ PVC 的属性，建立一对一绑定关系          │
└──────────────────────┬──────────────────────────┘
                       │  PVC 作为命名空间资源存在
                       ▼
┌─────────────────────────────────────────────────┐
│              应用开发者职责                        │
│                                                   │
│  声明 PVC：需要多大容量、什么访问模式                │
│  Pod 中引用 claimName，存储细节完全透明              │
└─────────────────────────────────────────────────┘
```

这个分层设计的价值体现在三个层面：

- **运维层面**：存储后端的变更（如从 NFS 迁移到 Ceph）只需修改 PV 或 StorageClass，应用 YAML 无需改动。
- **开发层面**：开发者只需声明需求，不需要学习 NFS 挂载参数或云硬盘类型。
- **调度层面**：PV/PVC 绑定与 Pod 调度形成协作，`WaitForFirstConsumer` 确保拓扑一致性。

理解 PV 和 PVC，就是理解 Kubernetes 如何将"存储"这个有状态世界的核心实体，优雅地接入无状态容器编排体系之中。下一层抽象是 StorageClass 和 CSI——它们分别解决了"谁来创建存储"和"不同的存储如何以统一接口接入"的问题，但这已经是另一篇文章的内容了。

---

**关键词**：PersistentVolume、PersistentVolumeClaim、PV、PVC、Kubernetes 存储、动态供给、静态供给、StorageClass、访问模式、回收策略、卷模式、WaitForFirstConsumer、本地 PV、块存储、CSI
