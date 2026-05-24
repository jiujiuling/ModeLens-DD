# ModeLens-DD 实验报告

## 1. 项目目标
本项目验证一个 mode-aware Rényi dataset distillation 原型：把真实分类数据压缩成少量可训练合成样本，同时尽量保持类别条件分布、多模态覆盖关系和特征空间熵结构。

## 2. 算法机制
核心流程是：先用冻结随机编码器池得到 identity 与多层非线性特征；再对每个类别、每个特征层建立 mode bank；最后直接优化合成点，使其在 Cauchy-Schwarz kernel divergence、mode coverage、matrix Rényi entropy 三类指标上接近真实数据。

## 3. 实验结果
程序已输出蒸馏前后分布图、loss curve、合成数据文件和本报告。若 loss 曲线下降且合成点覆盖多个真实模式，说明该 MVP 已完成从“理论损失”到“可运行优化闭环”的验证。

## 4. 主要风险
第一，toy data 不能直接证明图像任务有效，需要迁移到 CIFAR-100 / Tiny-ImageNet。第二，核带宽过小会导致局部过拟合，过大则会抹平多模态结构。第三，coverage loss 权重过高可能牺牲分类边界附近的判别性。

## 5. 下一步计划
接入随机 ConvNet bank 与冻结 ResNet-18 特征；把 mode bank 从二维点扩展到类条件图像特征；加入非反传 Sinkhorn plan 作为 pair-wise reweighting；让 LLM Agent 自动阅读日志并生成下一轮超参数搜索计划。