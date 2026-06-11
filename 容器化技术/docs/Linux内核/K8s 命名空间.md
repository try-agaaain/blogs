# 容器世界的“平行宇宙”：深入理解 containerd 的命名空间隔离

你是否遇到过这样的场景：一台机器上既运行着 Docker 守护进程，又运行着 Kubernetes 的 kubelet，但它们各自管理的容器却“鸡犬之声相闻，老死不相往来”。用 Docker 客户端列出的容器永远少一批，而 Kubernetes 看到的 Pod 里也完全没有 Docker 启动的那个服务的影子。这并非诡异的 bug，而是一个精心设计的隔离机制在悄悄发挥作用。要真正理解这种“彼此透明”背后的原理，我们需要从容器运行时这片江湖的演化说起，一路深入到 containerd 的设计精髓——命名空间。

## 一、从单体引擎到标准接口：容器运行时的解耦之路

早期的 Docker 像一把瑞士军刀，把镜像构建、容器生命周期管理、网络、存储等所有能力集于一个守护进程 `dockerd` 之中。这种单体架构虽然让开发者上手极快，却也带来了耦合过重、难以替换部件的问题。随着容器编排（尤其是 Kubernetes）的崛起，业界对更轻量、更专注的容器运行时需求愈发强烈。

为此，社区达成了两个关键标准：
- **OCI (Open Container Initiative)**：定义了容器镜像格式 (image-spec) 和容器运行时规范 (runtime-spec)，使得 `runc` 这类低层运行时只需遵循规范即可被上层任意调用。
- **CRI (Container Runtime Interface)**：Kubernetes 为解除对 Docker 的强依赖，抽象出的一套 gRPC 接口。任何实现了 `RuntimeService` 和 `ImageService` 的运行时组件，都可以成为 Kubernetes 的“容器引擎”。

在这两股力量推动下，Docker 自身也开启了组件化拆分。它把核心的容器管理逻辑剥离出来，形成了 **containerd** 这个独立的守护进程。containerd 向上提供管理镜像、容器的 gRPC API，向下则通过 `containerd-shim` 调用 `runc` 等 OCI 兼容运行时。这个捐赠给 CNCF 的项目，很快就成为了容器生态的事实标准中间层。

## 二、containerd 的架构与“多租户”基因

containerd 的设计从一开始就不是只为 Docker 一家服务。它的架构核心是一个强大的插件系统：内容存储 (content)、镜像服务 (images)、容器管理 (containers)、任务执行 (tasks)、快照器 (snapshots)……所有这些模块都以插件形式存在，并通过 gRPC 对外暴露。更重要的是，containerd 在数据模型中引入了一个与 Linux 内核的 namespace 思想一脉相承却又截然不同的概念：**Containerd Namespace**。

这个命名空间不是内核级别的隔离（如 PID namespace、network namespace），而是一种**纯逻辑层面的资源视图隔离**。可以把它理解为 containerd 内部为不同客户端开辟的独立工作区。每个工作区拥有自己的一套容器、镜像、快照和任务记录，彼此完全不可见。

从实现角度看，containerd 使用内嵌的 BoltDB 数据库存储元数据。当你创建一个容器时，这个容器的元信息会被写入数据库，而命名空间就是关键索引之一。客户端的每一次请求都需要在 context 中携带目标命名空间；containerd 在查询或操作数据时，会严格过滤出该命名空间下的条目。这意味着即便只有一个 containerd 进程、共用一个 `/run/containerd/containerd.sock` 套接字，它也能在同一片物理存储上支撑起多个完全隔离的逻辑视图。

## 三、Docker 的领地：moby 命名空间

当 containerd 从 Docker 中独立出来后，Docker 引擎自身也演变为一个更薄的上层封装。现代的 `dockerd` 启动时，会在内部启动一个 containerd 实例，或者直接连接到系统已有的 containerd。无论是哪种模式，Docker 与 containerd 交互时，都会明确指定一个特定的命名空间：**`moby`**。

这个命名空间名称承载着历史——Docker 公司的前身就叫 Moby Project。你在 Docker 中构建的镜像、运行的容器，其生命周期管理最终都会沉淀到 containerd 的 `moby` 命名空间下。当你通过 Docker CLI 请求容器列表时，`dockerd` 会向 containerd 发起调用，而 containerd 只会返回 `moby` 命名空间内的容器信息。对于 `default` 或其它命名空间里的东西，Docker 一无所知。

## 四、Kubernetes 的疆域：k8s.io 命名空间

Kubernetes 的情况要丰富一些。早年间，kubelet 通过一个叫 `dockershim` 的内部适配层把 CRI 请求翻译成 Docker API，因而容器最终还是落在 `moby` 命名空间下。但随着 `dockershim` 被弃用，现在主流的部署方式是将 containerd 作为 CRI 运行时直接对接。

containerd 内置了一个名为 `CRI` 的插件，这个插件完美实现了 Kubernetes 要求的 `RuntimeService` 和 `ImageService`。当 kubelet 通过 CRI gRPC 调用 containerd 时，CRI 插件便会接管请求，并将所有由 Kubernetes 管理的 Pod 的容器、沙箱（pause 容器）等资源，整齐地放进另一个专用的命名空间：**`k8s.io`**。

于是，一条清晰的分界线形成了：Docker 在 `moby` 下开垦，Kubernetes 在 `k8s.io` 下播种。即便二者使用同一个 containerd 进程和同一个 socket，它们的指令也会被命名空间这条“国界”严格导向各自的辖区。

## 五、共享而不混淆：单机共存的奥秘

现在我们可以解释文章开头的现象了。在一台节点上，如果你为了让开发者能便捷地构建镜像而保留了 Docker，同时又安装了 Kubernetes 并使用 containerd 作为 CRI 运行时，那么实际的架构很可能如下：

1. 系统运行着一个 containerd 服务，套接字为 `/run/containerd/containerd.sock`。
2. `dockerd` 启动时连接该 containerd，并开始使用 `moby` 命名空间。你通过 Docker 运行一个测试容器，其元数据便写入了 `moby` 之下。
3. kubelet 同时连接同一个 containerd socket，但其 CRI 插件强制使用 `k8s.io` 命名空间。当 Kubernetes 调度 Pod 到本节点时，对应的容器和沙箱会写入 `k8s.io`。

此时，containerd 内部实际管理着两组完全隔离的容器集合。Docker 视角下没有任何 Kubernetes 的痕迹，因为它只被授权查看 `moby` 命名空间；同样，kubelet 也不会看到 Docker 自行启动的容器，因为它的世界仅限于 `k8s.io`。你甚至可以用 `ctr --namespace moby containers ls` 和 `crictl ps`（底层连接 `k8s.io`）这类工具来直接验证两个命名空间的并存。

## 六、设计哲学：隔离、解耦与演进能力

containerd 的命名空间机制，本质上是对“关注点分离”这一软件工程原则的绝佳实践。它让多个上层系统可以安全地共享同一个底层运行时，而不必担心资源冲突或元数据污染。这种设计也使得 containerd 能够同时服务不同的消费者：除了 Docker 和 Kubernetes，还可以有云厂商自研的编排器、Serverless 容器平台，甚至简单的调试工具，它们只需申请一个属于自己的命名空间，便可在容器世界里划地自治。

更深一层看，这种逻辑隔离恰恰是容器生态标准化过程中的必然产物。当运行时与编排器解耦、当 Docker 不再是一站式入口，我们就需要一种轻量级的“多租户”能力来承载多样化的上层需求。命名空间模式，在优雅性和实现成本之间找到了最佳平衡点。

理解这一点后，再回头看那些“容器无故消失”或“磁盘莫名多出镜像”的怪异现象，往往就有了清晰的排查路线：你可能误入了别人的命名空间，或者某个客户端没有使用预期的命名空间。在现代基础设施的拼图里，containerd 的命名空间就像一张透明的分层画布，每一层都运行着一个独立的故事，而它们共同依赖的，不过是同一颗跳动的心脏。