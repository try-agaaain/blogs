# 容器技术深度解析：Docker、containerd 与 OCI

从“One Docker”到开放生态，容器世界早已不是 Docker 的一言堂。要真正理解现代容器基础设施，就必须回到那个分裂与标准化并存的时刻，看清 **Docker**、**containerd** 与 **OCI** 三者如何各自演化、又如何咬合在一起。

---

## 1. 从 Docker 说起：体验的胜利

2013 年，dotCloud 公司开源了 Docker。一句 `docker run -d nginx` 把 Linux 内核中零碎的 namespace、cgroup、联合文件系统瞬间打包成可复现的应用沙箱。开发者第一次感受到：

- 我的机器上能跑，服务器上也能跑。
- 交付的不再是 war 包，而是一个自包含的“标准集装箱”。

那个时代的 Docker 是一个**单体应用**，Docker Daemon 包揽了从镜像拉取、容器创建、日志收集到网络编排的全部工作。内部依赖链大致是：

```
CLI → Docker Daemon →  containerd (Docker 自研的早期版本)  →  libcontainer → Linux 内核
```

这里有个关键信号：**libcontainer**。早期 Docker 用 LXC 作为默认执行驱动，但很快 Docker 团队发现 LXC 抽象臃肿、跨发行版适配难，于是自己用 Go 写了 libcontainer 直接操作内核。2015 年，Docker 将 libcontainer 捐赠给 OCI（Open Container Initiative），这就是 `runc` 的前身。从那一刻起，标准化的齿轮开始转动。

---

## 2. 容器世界的“宪法”——OCI

OCI 是 Linux 基金会下的开放容器标准组织，使命是**让容器运行时和镜像格式不再被任何厂商绑定**。它包含三个核心规范：

- **Runtime Specification（运行时规范）**：定义如何从一份文件系统包（rootfs）和一份 `config.json` 启动一个容器。符合该规范的运行时称为 OCI Runtime，比如 `runc`、`crun`、`kata-runtime`。只要给出符合规范的文件系统，任何 OCI Runtime 都能跑。
- **Image Specification（镜像规范）**：定义镜像的层、配置、manifest、索引等格式。你可以用 Docker 构建镜像，用 Podman 推送，用 containerd 拉取——只要都遵守 OCI Image Spec。
- **Distribution Specification（分发规范）**：定义镜像仓库的 API，推动 registry 的互操作性。

OCI 的出现使容器运行时百花齐放：`runc` 是默认的低级运行时，`kata-runtime` 把容器跑在轻量虚拟机里，`gVisor` 用用户态内核拦截系统调用。而这一切，都源于 Docker 当年那个“剥离 libcontainer”的瞬间。

---

## 3. 拆分：Docker 化身为组件

单体 Docker Daemon 的劣势逐渐暴露：升级内核、重启守护进程会导致所有容器挂掉；难以被 Kubernetes 这类编排系统直接调度。Kubernetes 更不想绑定一个庞大的 Docker，于是在 v1.20 宣布弃用 Docker 作为容器运行时（虽然后来澄清只是弃用 `dockershim`）。

Docker 的应对是**拆分自身**：把容器运行时逻辑独立为 **containerd**，并于 2017 年捐赠给 CNCF。此时的架构变成：

```
Docker CLI → Docker Daemon → containerd → containerd-shim → runc → 容器进程
```

现在，用户仍可享受 `docker build`、`docker compose` 等开发体验，而 containerd 成为上游基础设施的公共层。Kubernetes 可以直接通过 CRI 对接 containerd，绕开 Docker Daemon。

---

## 4. containerd：云原生容器管家

containerd 的定位是**高级容器运行时**，它不直接创建容器（那是 runc 的事），而是管理容器的整个生命周期，同时处理镜像、存储、网络、执行等复杂事务。

containerd 通过 gRPC API 对外暴露服务，内部采用**插件化架构**，每个领域都是一个插件：

```
Content → Snapshots → Images → Containers → Tasks → Events ...
```

其中：

- **Content 插件**：管理镜像层的原始 blob。
- **Snapshot 插件**：将不同镜像层通过 overlay、btrfs 等快照器组合成容器的 rootfs。
- **Image 插件**：负责镜像的拉取、解压、挂载。
- **Container 与 Task 插件**：Container 是静态配置（如命令、挂载点），Task 是运行中的容器实例。
- **shim**：containerd 为每个容器启动一个 shim 进程，即使 containerd 自己重启，shim 也保持容器存活，并通过 ttrpc 与 containerd 重新连接。这解决了 Docker Daemon 重启杀容器的顽疾。

### containerd 与 Kubernetes

Kuberentes 通过 CRI (Container Runtime Interface) 定义了对容器运行时的需求。containerd 内置了 CRI 插件，使得 kubelet 可以直接通过 Unix socket 调用 containerd，无需 `dockershim`。调用链简化为：

```
kubelet (CRI) → containerd → runc
```

相比 Docker 模式，少了一层 docker daemon，减少了内存开销和调用延迟，也降低了系统复杂度。

---

## 5. runc：那个真正创建容器的“低级运行时”

OCI Runtime 规范的核心实现就是 **runc**。它的工作非常纯粹：读一份符合 OCI 规范的 `config.json`，然后调用 Linux 的 `clone()`、`setns()`、`unshare()` 等系统调用，为进程创建独立的 namespace，并通过 cgroup 限制资源。

创建一个容器的过程可以简化为：

```bash
# 准备好 rootfs 和 config.json 后
runc run mycontainer
```

runc 自己就是容器的父进程，没有额外的守护进程。这与 containerd 的 shim 不同：shim 负责保持 STDIO 通道和报告退出状态，runc 只负责“起”的那一瞬间。容器起来后 runc 就退出，剩下的由 shim 接管。

runc 代码库小巧精悍，是学习 Linux 容器原理的绝佳材料。它也衍生出安全增强版本，如 `crun`（C 语言编写，更快更轻）和 `kata-runtime`（替换 runc 的过程为虚拟机启动）。

---

## 6. 完整调用链：当你在 Kubernetes 中创建一个 Pod

假设集群的容器运行时是 containerd：

1. **kubelet** 收到调度 Pod 的指令。
2. kubelet 通过 CRI gRPC 调用 `RunPodSandbox`，containerd 创建 **Pod 沙箱**（通常是pause容器，共享网络命名空间）。
3. kubelet 调用 `CreateContainer` 和 `StartContainer`，containerd 检查镜像是否已在本地，若没有则通过 content/snapshot 插件拉取并解压镜像。
4. containerd 生成 OCI 规范所需的 `config.json`，并调用 **containerd-shim** 启动新子进程。
5. shim 再调用 `runc create` + `runc start` 创建容器进程，并把 stdout/stderr 回流给 containerd。
6. 容器运行。containerd 退出也不影响，shim 一直守护，直到容器终止并报告退出码。

整个过程中，Docker 完全没有出现。这也是为什么 “Kubernetes 弃用 Docker” 的核心原因——Kubernetes 要的是 containerd，而不是那层 Docker Daemon。

---

## 7. 容器运行时的生态地图

如今，CNCF 景观下的容器运行时已三分天下：

| 运行时                                  | 层级             | 特点                                                         |
| --------------------------------------- | ---------------- | ------------------------------------------------------------ |
| **containerd**                          | 高级             | CNCF 毕业项目，Docker 及多数 K8s 发行版的默认运行时，插件丰富 |
| **CRI-O**                               | 高级             | 专为 Kubernetes CRI 打造，更轻量，无多余特性，Red Hat 推动   |
| **runc**                                | 低级             | OCI 参考实现，Go 编写，被所有高级运行时依赖                  |
| **crun**                                | 低级             | C 实现，内存和启动速度更优，支持 cgroup v2 等新特性          |
| **kata-runtime / gVisor / Firecracker** | 低级（安全容器） | 提供更强的隔离，常用于多租户和不可信负载                     |

Docker 自身也演变为构建工具与桌面体验的集合，其底层依然是 containerd。在开发者工作站上，`docker build` 的无缝体验依然无可替代；在服务器上，containerd 与 CRI-O 的无守护进程重启特性和原生 CRI 支持更符合生产需求。

---

## 8. 走向更深层的标准化与轻量化

容器技术正沿着两条道路前进：

- **接口标准化**：OCI 规范已稳定，未来可能在镜像分发、签名、漏洞信息等领域扩展。而 Kubernetes 的 CRI 正在向 gRPC 标准化方向持续演进。
- **运行时瘦身与安全**：以 `crun`、`youki`（Rust 实现）为代表的轻型 OCI 运行时追求更低开销，安全容器则借助虚拟化技术提供硬件级隔离。两者都依赖 OCI 规范来保证与上层 containerd/CRI-O 的解耦。

理解 Docker、containerd 与 OCI 的纠缠，实际上是在理解容器技术如何从单一产品走向开放平台。下一次你执行 `docker run` 或 `crictl pull` 时，或许会清晰地看见那个由标准、插件和 shim 织成的隐形世界——它安静，却支撑着今天所有的云原生应用。

---

**关键词**：容器运行时、Docker、containerd、OCI、runc、CRI、Kubernetes、namespaces、cgroups、镜像规范、shim