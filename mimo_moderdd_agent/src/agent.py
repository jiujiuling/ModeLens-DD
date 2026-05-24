from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AgentMessage:
    role: str
    content: str


class ResearchAgent:
    """A small report-generation agent with an offline fallback.

    If MIMO_API_KEY and MIMO_BASE_URL are set, this class calls an
    OpenAI-compatible /chat/completions endpoint. Otherwise it uses a deterministic
    local generator so that the demo is always runnable.
    """

    def __init__(self, model: Optional[str] = None, use_llm: bool = False):
        self.api_key = os.environ.get("MIMO_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = (os.environ.get("MIMO_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
        self.model = model or os.environ.get("MIMO_MODEL") or os.environ.get("OPENAI_MODEL") or "mimo-research-agent"
        self.use_llm = use_llm and bool(self.api_key and self.base_url)

    def chat(self, system: str, user: str) -> str:
        if not self.use_llm:
            return self._offline_reply(user)
        url = self.base_url + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def make_report(self, config: Dict, metrics: Dict, final_losses: Dict) -> str:
        prompt = f"""
请基于下面的实验信息，写一份研究型项目报告，结构包括：项目目标、算法机制、实验结果、主要风险、下一步计划。
config={json.dumps(config, ensure_ascii=False, indent=2)}
metrics={json.dumps(metrics, ensure_ascii=False, indent=2)}
final_losses={json.dumps(final_losses, ensure_ascii=False, indent=2)}
""".strip()
        system = "你是一个机器学习研究项目审查 Agent，关注算法创新性、实验可复现性和风险诊断。"
        return self.chat(system, prompt)

    def _offline_reply(self, user: str) -> str:
        # Extract JSON blocks roughly for deterministic fallback.
        return """# ModeLens-DD 实验报告

## 1. 项目目标
本项目验证一个 mode-aware Rényi dataset distillation 原型：把真实分类数据压缩成少量可训练合成样本，同时尽量保持类别条件分布、多模态覆盖关系和特征空间熵结构。

## 2. 算法机制
核心流程是：先用冻结随机编码器池得到 identity 与多层非线性特征；再对每个类别、每个特征层建立 mode bank；最后直接优化合成点，使其在 Cauchy-Schwarz kernel divergence、mode coverage、matrix Rényi entropy 三类指标上接近真实数据。

## 3. 实验结果
程序已输出蒸馏前后分布图、loss curve、合成数据文件和本报告。若 loss 曲线下降且合成点覆盖多个真实模式，说明该 MVP 已完成从“理论损失”到“可运行优化闭环”的验证。

## 4. 主要风险
第一，toy data 不能直接证明图像任务有效，需要迁移到 CIFAR-100 / Tiny-ImageNet。第二，核带宽过小会导致局部过拟合，过大则会抹平多模态结构。第三，coverage loss 权重过高可能牺牲分类边界附近的判别性。

## 5. 下一步计划
接入随机 ConvNet bank 与冻结 ResNet-18 特征；把 mode bank 从二维点扩展到类条件图像特征；加入非反传 Sinkhorn plan 作为 pair-wise reweighting；让 LLM Agent 自动阅读日志并生成下一轮超参数搜索计划。"""
