# Book Principles

## 1. Book Type

技术书 + 实战教程（technical / tutorial）。采用「技术书章节」模板，每章必须有完整可运行代码。

## 2. Target Reader

有编程基础的工程师（会 Python、会命令行），不要求深度学习背景。

读者卡点：听说过 LLM 训练的各种名词，但从未亲手跑通过任何一环。

读者读完应获得：独立完成"准备数据 → 训练/微调 → 评估 → 部署"全流程的能力，以及对每个环节背后机制的判断力。

## 3. Core Goal

让读者**亲手跑通 LLM 全流程**：手写 Transformer → 预训练 → SFT → LoRA/QLoRA → DPO → 评估 → 多模态 → 部署。

学习哲学：「手写一遍再上工具」——先用 PyTorch 从零实现理解原理，再用 HuggingFace 生态工程化复现同一件事。

## 4. Core Principles（本书特化，硬性约束）

1. **代码可运行是底线**：每章代码必须完整、可独立运行、无伪代码、无 `...` 省略关键逻辑。每章开头标注「环境要求」块：Python 版本、依赖及版本、内存/显存需求、预计运行时间。
2. **小数据集原则**：全书只用 KB~MB 级数据集。数据要麽手写构造、要麽给出生成脚本、要麽给出可直接下载的小规模抽样。目的是理解流程，不是刷分。
3. **低算力优先**：默认环境为 CPU / Mac(MPS)。每章代码顶部统一使用 `DEVICE` 自动检测（cuda > mps > cpu）。需要 GPU 的实验必须给出三档方案：Mac 本地小模型 / Colab 免费 / AutoDL 按小时租用（附完整租用与 SSH 操作流程）。
4. **单文件脚本**：每章代码以单个 `.py` 或单个 Notebook 形式给出，读者复制即可跑，不依赖本书之外的私有模块。跨章复用的小组件在引入章完整给出。
5. **先问题后机制**：每个模块先说"它解决什么问题"，再给实现。禁止无铺垫的术语轰炸。
6. **双轨对照**：同一任务尽量给出「从零实现」和「库实现」两个版本，并指出两者对应关系与库的额外便利。
7. **可观察的训练**：所有训练代码必须打印/记录 loss 曲线，并在正文解读曲线形态（正常/过拟合/学习率不对长什么样）。
8. **随机性要固定**：所有训练代码设置随机种子，书中给出的示例输出必须是真实运行结果（标注运行环境）。

## 5. Writing Style

- 语气直接、准确、务实，像资深同事结对编程时的讲解
- 每章 3000–5000 字 + 代码；不为凑字数扩写
- 中文正文，关键术语首次出现补英文括号，如：低秩适配（Low-Rank Adaptation, LoRA）
- 代码注释克制：解释"为什么"而非复述代码
- 每章至少一个可运行的完整例子；技术章节必须有完整端到端例子
- 输出示例用真实运行结果，标注运行硬件（如 `# Mac M1 / CPU 输出：`）

## 6. Required Structure（技术书章节模板）

```text
## 本章要解决的问题
## 环境要求（依赖版本 + 硬件 + 预计运行时间）
## 核心概念
## 实现路径（分块讲解，每块：问题 → 原理 → 代码）
## 完整例子（端到端可运行）
## 结果解读（输出/loss 曲线长什么样，怎么判断对错）
## 边界与取舍（什么时候该用别的方案）
## 本章产物（读者拿到了什么可复用文件/脚本/判断力）
## 小结：连接下一章
```

## 7. Forbidden Patterns

- 禁止伪代码和省略号式代码（`# ... 其余同上`）
- 禁止百科式概念罗列（如大而全的 Transformer 变体综述）
- 禁止模型发展史堆叠（GPT-1 到 GPT-4 的编年史叙事）
- 禁止"本章介绍了……"式开头（要用读者问题开头）
- 禁止无法在小数据集上验证的示例
- 禁止未标注硬件需求的大模型实验
- 禁止制造引用、虚构 benchmark 数字

## 8. 概念—操作能力原则（Concept–Operation Capability）

本书的增益度量框架。读者长期内化的不是具体语法和 API，而是本领域的 **Concept–Operation Map**：

> **看到问题时，知道有哪些 Concept 可以调用；**
> **看到 Concept 时，知道有哪些 Operation 可以施加。**

- 概念是能力的地址（Concept is the address of capability）——不知道概念存在，就无法主动调用围绕它的能力
- 操作是能力的杠杆（Operation is the leverage of capability）——Operation 告诉读者哪里可以动
- 掌握层级链：**Name → Concept → Operation → Relation → Judgment**。本书的目标是把读者从"听说过"推进到"知道怎么做、和什么相互影响、何时该用"

**每章末尾的"本章产物"必须包含一小节「概念—操作增量」**，用三行明确回答：

| 增量 | 回答的问题 |
|---|---|
| Concept 增量 | 本章让读者"知道了哪些此前不知道的存在/边界" |
| Operation 增量 | 读者现在"能对它们做什么"（可执行动作，含判断何时用） |
| Judgment 增量 | "什么时候用、什么时候不用、什么叫好"的判断规则 |

**实现细节的定位**：本书是动手教程，代码是章节产物的一部分，但正文主线承载的是概念、操作与判断——具体函数签名/参数细节保留在 `code/` 目录的脚本注释中（按需查询），正文只讲"为什么、何时、怎么判断"。
**关键概念的最小知识单元**：Concept / Purpose / Operations / Relations / When-Limits / Evaluation 六字段。本书通过"核心概念 + 边界与取舍 + 结果解读"三个小节组合覆盖，不单独列六字段表（避免版式重复）。

## 8. Terminology

| 中文 | English | 缩写 | 备注 |
|---|---|---|---|
| 语言模型 | Language Model | LM | |
| 大语言模型 | Large Language Model | LLM | |
| 分词器 | Tokenizer | | |
| 字节对编码 | Byte Pair Encoding | BPE | |
| 词表 | Vocabulary | | |
| 预训练 | Pretraining | | |
| 监督微调 | Supervised Fine-Tuning | SFT | |
| 指令微调 | Instruction Tuning | | SFT 的同位语 |
| 低秩适配 | Low-Rank Adaptation | LoRA | |
| 量化 | Quantization | | |
| 嵌入 | Embedding | | |
| 注意力 | Attention | | |
| 自注意力 | Self-Attention | | |
| 多头注意力 | Multi-Head Attention | MHA | |
| 因果掩码 | Causal Mask | | |
| 前馈网络 | Feed-Forward Network | FFN | |
| 残差连接 | Residual Connection | | |
| 人类反馈强化学习 | RLHF | RLHF | 只作概念 |
| 直接偏好优化 | Direct Preference Optimization | DPO | |
| 奖励模型 | Reward Model | RM | |
| 参考模型 | Reference Model | | |
| 困惑度 | Perplexity | PPL | |
| 梯度检查点 | Gradient Checkpointing | | |
| 混合精度 | Mixed Precision | | |
| 检查点 | Checkpoint | | 模型存档，避免与梯度检查点混淆 |
| 视觉语言模型 | Vision-Language Model | VLM | |
| 上下文窗口 | Context Window | | |
| 温度采样 | Temperature Sampling | | |
