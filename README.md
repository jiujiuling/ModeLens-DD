ModeLens-DD: Agentic Mode-aware Rényi Dataset Distillation Workbench
这是一个可直接运行的研究型 MVP：用 模式感知 Rényi / Cauchy-Schwarz 分布匹配 把一个真实二维分类数据集蒸馏成极少量合成样本，同时由一个轻量 Research Agent 生成实验计划、风险审查和项目报告。项目适合用于申请大模型 token 计划：它不是普通聊天应用，而是把 LLM/Agent 用在“研究假设 → 损失函数设计 → 实验诊断 → 报告生成”的闭环里。

技术点
Class-conditional dataset distillation：每个类别独立优化合成样本。
Frozen random encoder bank：用多层随机特征空间替代单一输入空间匹配。
Mode bank：对真实特征按类别、按层做 KMeans，保存模式中心和模式占比。
Rényi/Cauchy-Schwarz kernel divergence：用核密度内积近似两个分布的差异。
Mode coverage regularizer：约束合成样本覆盖真实分布的多模态结构。
Matrix Rényi entropy matching：用归一化 Gram 矩阵估计二阶 Rényi 熵，抑制合成样本坍缩。
Agentic report loop：可选调用 OpenAI-compatible API；无 API key 时使用 deterministic local agent fallback。
一键运行
cd mimo_moderdd_agent
pip install -r requirements.txt
python run_demo.py --epochs 200 --ipc 8 --dataset moons --out outputs/demo
运行后会生成：

outputs/demo/synthetic_before.png
outputs/demo/synthetic_after.png
outputs/demo/loss_curve.png
outputs/demo/report.md
outputs/demo/synthetic_data.npz
可接入 MiMo / OpenAI-compatible API
如果你的模型服务兼容 /v1/chat/completions，可以设置：

export MIMO_API_KEY="你的_key"
export MIMO_BASE_URL="https://your-endpoint/v1"
export MIMO_MODEL="your-model-name"
python run_demo.py --epochs 200 --ipc 8 --use-llm-agent
没有这些环境变量时，项目仍然能离线运行。

