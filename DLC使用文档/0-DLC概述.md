手动配置深度学习训练集群耗时长、易出错。分布式训练DLC（Deep Learning Containers）基于Kubernetes提供开箱即用的训练环境，无需配置即可快速启动训练任务。支持多种深度学习框架和灵骏智算、GPU等算力资源。

## 产品优势

-   **支持多样算力资源**
    
    基于灵骏智算和通用计算资源，支持云上ECS、ECI、神龙裸金属和灵骏裸金属等多种算力形态，实现异构算力的混合调度。
    
-   **多样的分布式任务类型**
    
    无需搭建集群，直接提交Megatron、Deepspeed、Pytorch、Tensorflow、MPI及XGBoost等十多种训练框架的任务。DLC预置多种官方镜像，也支持自定义运行环境，可通过控制台、SDK或命令行提交任务。
    
-   **高稳定**
    
    在大模型训练场景中，DLC通过自研的容错引擎AIMaster、高性能Checkpoint框架EasyCKPT、健康检测SanityCheck和节点自愈功能，自动检测和恢复故障，降低算力损失，提升训练稳定性。
    
-   **高性能**
    
    自研AI训练加速框架支持数据并行、流水并行、算子拆分等多种并行策略，并提供并行策略自动探索和显存优化能力。结合拓扑感知调度、梯度分组融合、混合精度通信等通信优化手段，提升分布式训练效率。适用于大模型预训练、持续训练和Alignment等场景。
    

## **应用场景**

-   **数据预处理**：支持自定义运行环境，对数据进行离线并行预处理，降低数据预处理难度。
    
-   **大规模分布式训练**：支持多种开源深度学习框架，可使用上千个节点同时训练，显著缩短训练时间。
    
-   **离线推理**：支持使用DLC进行离线推理，提升闲时GPU资源利用率。
    

## 资源形态

PAI提供以下两种资源形态，您可根据算力需求选择：

-   **灵骏智算**：专为大模型训练设计的高性能算力资源，支持超大规模深度学习任务。适用于大模型训练、自动驾驶、基础科研等需要大量计算资源的场景。
    
-   **通用计算**：适用于常规训练需求，能够灵活地支持多种规模和类型的机器学习任务。
    

灵骏智算和通用计算资源支持以下几种使用方式：

-   **资源配额**：通过**包年包月**提前购买灵骏智算或通用计算资源，适合长期稳定的训练需求。
    
-   **公共资源**：无需提前购买资源，提交任务时按需使用灵骏智算或通用计算资源，按量付费结算。
    
-   **竞价资源**：灵骏智算提供竞价资源，帮助您以较低成本获取AI算力。
    

## 核心功能

### **任务创建与管理**

-   [创建训练任务](https://help.aliyun.com/zh/pai/create-a-training-task)：通过控制台、SDK或命令行提交训练任务，配置镜像、数据集、资源规格和启动命令等。
    
-   [查看训练任务详情](https://help.aliyun.com/zh/pai/view-training-jobs)：查看任务基本信息、资源视图和操作日志。
    
-   [管理训练任务](https://help.aliyun.com/zh/pai/manage-training-jobs)：停止、克隆、分享、生成脚本和删除任务。
    
-   [配置离线调度](https://help.aliyun.com/zh/pai/configure-automatic-job-submission-at-specific-points-in-time)：当数据或超参数更新时，定期提交DLC任务进行增量训练和模型调优。
    
-   [存储配置](https://help.aliyun.com/zh/pai/use-cloud-storage-for-a-dlc-job)：通过代码配置或挂载方式使用OSS、NAS、CPFS 存储。
    

### **任务监控与通知**

-   [查看训练任务分析报告](https://help.aliyun.com/zh/pai/use-tensorboard-to-view-training-results-in-dlc)：使用TensorBoard可视化训练指标和分析报告。
    
-   [使用云监控或ARMS监控训练任务](https://help.aliyun.com/zh/pai/training-monitoring-and-alerting)：查看DLC任务的资源状况或配置告警规则。
    
-   [配置消息通知](https://help.aliyun.com/zh/pai/configure-notification-on-job-status-changes)：在PAI工作空间的事件中心创建消息通知规则，实时跟踪任务状态变化。
    

### **网络与通信加速**

-   [RDMA配置](https://help.aliyun.com/zh/pai/use-a-custom-image)：使用灵骏智算资源时，配置高性能RDMA网络加速节点间通信。
    
-   [eRDMA](https://help.aliyun.com/zh/pai/erdma-distributed-training-with-high-performance-networks)：通用计算资源的部分GPU机型支持弹性RDMA，系统自动挂载网卡加速训练。
    
-   [ACCL：阿里云高性能集合通信库](https://help.aliyun.com/zh/pai/accl-alibaba-high-performance-collective-communication-library)：结合阿里云网络特点和大模型通信调优经验，提供高性能集合通信，并具备故障诊断和自愈能力。
    
-   [通过专有网关提升公网访问速率](https://help.aliyun.com/zh/pai/configure-a-dlc-instance-to-access-the-internet-through-a-private-nat-gateway)：为实例所在VPC创建公网NAT网关并配置SNAT，让实例通过专有公网网关高速访问互联网。
    

### **成本优化**

-   [竞价资源](https://help.aliyun.com/zh/pai/use-preemptible-job)：使用灵骏智算竞价资源创建DLC任务，以较低成本获取AI算力。
    
-   [使用闲时资源](https://help.aliyun.com/zh/pai/use-idle-resources-in-dlc-jobs)：提交闲时计算任务，利用其他Quota的空闲算力，在不影响正常业务的前提下提升资源利用率、降低训练成本。
    

### **弹性与容错**

-   [自动容错](https://help.aliyun.com/zh/pai/aimaster-elastic-fault-tolerant-engine)：通过AIMaster监控任务运行状态，自动检测并恢复故障。
    
-   [健康检测](https://help.aliyun.com/zh/pai/sanity-check)：训练前检测资源健康状况，自动隔离故障节点。
    
-   [EasyCkpt](https://help.aliyun.com/zh/pai/easyckpt)：为PyTorch大模型训练提供无损的模型保存和恢复能力，支持断点续训。
    
-   [PerfTracker：在线性能分析诊断工具](https://help.aliyun.com/zh/pai/perftracker-online-performance-analysis-diagnostic-tool)：在线采集各Worker的CUDA核函数、Python函数执行记录及硬件监控数据，自动生成分析报告，定位慢节点、瓶颈函数和Hang问题。
    

## **计费说明**

DLC主要对计算资源计费：

| 计算资源类型 | 计费方式 | 计费主体 | 计费规则 | 停止计费 |
| 公共资源 | 按量计费 | DLC任务运行时长（占用公共资源的时长）。 | `账单金额=节点数量×（单价/60）×运行时长（分钟）` | - DLC任务执行结束。 - DLC任务状态为已停止。 |
| AI计算资源（通用计算资源和灵骏智算资源） | 包年包月 | 详情请参见[AI计算资源计费说明](https://help.aliyun.com/zh/pai/product-overview/ai-computing-resource-billing-description)。 | 详情请参见[AI计算资源计费说明](https://help.aliyun.com/zh/pai/product-overview/ai-computing-resource-billing-description)。 | 不涉及 |

更多详情请参见[分布式训练（DLC）计费说明](https://help.aliyun.com/zh/pai/product-overview/billing-of-dlc)。

## **快速入门**

以 MNIST 手写体识别为案例，为您介绍使用DLC进行单机单卡训练，或多机多卡的分布式训练，详情请参见[分布式训练 DLC 快速入门](https://help.aliyun.com/zh/pai/product-overview/distributed-training-dlc-quickstart)。

## **获取帮助**

-   **查阅DLC常见问题**：如果您遇到任务启动/停止失败、扣费疑问、训练报错等问题，请查阅[DLC常见问题](https://help.aliyun.com/zh/pai/faq-about-dlc)。
    
-   **询问PAI智能助手（Agentic PAI）**：在PAI控制台页面右上角单击![image](https://help-static-aliyun-doc.aliyuncs.com/assets/img/zh-CN/5893169871/p1102710.png)图标唤起PAI智能助手（Agentic PAI）。Agentic PAI 能提供PAI全链路产品的使用说明和操作指引，支持对DSW实例、DLC任务、EAS服务进行运维诊断，自动识别失败原因并给出处理建议。
    

## **相关文档**

-   [创建训练任务](https://help.aliyun.com/zh/pai/create-a-training-task)：通过控制台、SDK或命令行提交训练任务，以及了解其中关键参数如何配置。
    
-   [DLC使用案例汇总](https://help.aliyun.com/zh/pai/dlc-use-case-summary)：通过实际案例，来了解如何使用DLC。