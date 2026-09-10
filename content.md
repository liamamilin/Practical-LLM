# Book Title: 动手做大模型：从零训练到微调实战

## Book Metadata

- title: 动手做大模型
- subtitle: 从零训练到微调实战
- author: TODO
- book_type: technical / tutorial（技术书 + 实战教程）
- target_reader: 有编程基础的工程师，不要求深度学习背景
- core_goal: 读者亲手跑通 LLM 全流程——从手写 Transformer 到预训练、SFT、LoRA/QLoRA、DPO 对齐、评估、多模态与部署，建立可迁移的工程判断力
- writing_style: 准确、可执行；每章都有完整可运行代码；小数据集、低算力优先
- hardware_policy: 主线代码在 CPU / Mac(MPS) 本地可跑；需要 GPU 的实验（第 7 章等）提供 AutoDL / Colab 租用方案，成本控制在每小时几元钱
- dataset_policy: 全书只用小型数据集（KB~MB 级），目的是理解流程而非刷 benchmark

## Part 1：原理与从零实现

### Chapter 1：大模型全景——LLM 到底是什么、怎么练成的

- goal: 建立 LLM 全流程地图：预训练 → SFT → 对齐 → 部署；理解"从零手写"和"工具实战"两条学习路线在全书中的位置
- reader_problem: 听过预训练、微调、RLHF 等词但说不清它们的关系和顺序；不知道学 LLM 应该从哪里下手
- key_concepts: 语言模型（Language Model）、token、参数量、预训练（Pretraining）、监督微调（SFT）、对齐（Alignment）、推理（Inference）
- expected_output: 一张读者自己能画出的 LLM 训练全流程图；本章运行一个最小推理示例（加载小模型生成文本）
- notes: 不写模型发展史；直接按"数据流"讲全景

### Chapter 2：Tokenizer——文本如何变成数字

- goal: 理解 token 的本质；亲手实现 BPE（Byte Pair Encoding）；会用 HuggingFace tokenizer 验证
- reader_problem: 不理解 max_length、词表大小、`<|endoftext|>` 这些概念；不明白为什么模型算 token 而不算字
- key_concepts: 字符级/词级/子词分词、BPE、词表（Vocabulary）、特殊 token、字节级编码
- expected_output: 一个可运行的 mini-BPE 实现（200 行以内）+ 与 HuggingFace `tokenizers` 库结果的对照
- notes: 这是全书第一个"从零手写"完整闭环的章节

### Chapter 3：Attention 与 Transformer——手写一个 mini-GPT

- goal: 从 Self-Attention 写起，逐块搭出 Transformer（embedding、attention、FFN、残差、LayerNorm），最终组装成可训练的 mini-GPT
- reader_problem: 看过无数张 Transformer 结构图但从未写过一行；对 Q/K/V、多头、因果掩码只知其名
- key_concepts: Self-Attention、Q/K/V、多头注意力（Multi-Head Attention）、因果掩码（Causal Mask）、位置编码、残差连接、LayerNorm、前馈网络（FFN）
- expected_output: 一个完整可运行的 mini-GPT（数百万参数级），能对一个 toy 语料做 forward 和 loss 计算；CPU/MPS 上秒级运行
- notes: 代码逐块讲解，每块先给"它解决什么问题"再给实现

### Chapter 4：预训练——让模型学会说话

- goal: 在小型语料（如 TinyStories 中文子集或自建小语料）上完成一次完整的预训练：数据准备、训练循环、loss 曲线、采样生成
- reader_problem: 不清楚预训练到底在"学"什么；没见过训练循环长什么样；不理解 loss、学习率、warmup 这些训练参数
- key_concepts: 训练循环（Training Loop）、交叉熵损失、学习率调度（LR schedule）、warmup、梯度裁剪、上下文窗口、温度（Temperature）/top-k 采样
- expected_output: 一个能在 CPU/MPS 几分钟内训完的预训练脚本 + 生成的"能说人话"的文本样本；给出 GPU 上的对照实验
- notes: 这是 Part 1 的收束章节，把第 2、3 章的产物串成完整 pipeline

## Part 2：微调实战

### Chapter 5：指令微调 SFT——从"补全"到"听话"

- goal: 理解预训练模型与指令模型的差异；构造小型指令数据集；对小模型做全参数 SFT
- reader_problem: 不理解为什么预训练模型不会回答问题、只会续写；不知道 SFT 数据长什么样、训练代码怎么写
- key_concepts: 指令微调（Supervised Fine-Tuning）、对话模板（Chat Template）、prompt/completion、数据掩码（只对 response 计算 loss）、过拟合与欠拟合
- expected_output: 一个几百条的小型指令数据集 + 全参 SFT 脚本 + 微调前后行为对比示例
- notes: 数据集用手写构造 + 公开小数据集（如 alpaca-cleaned 抽样）混合

### Chapter 6：参数高效微调 PEFT——手写 LoRA

- goal: 理解全参微调的显存瓶颈；手写 LoRA 核心逻辑（低秩分解、合并、只训增量）；用 HuggingFace PEFT 复现同一实验
- reader_problem: 听过 LoRA 但说不清 rank/alpha 的含义；不理解为什么冻结大矩阵只训小矩阵就有效
- key_concepts: 全参微调显存分析、低秩适配（Low-Rank Adaptation）、rank（r）、缩放因子（alpha）、目标模块（target_modules）、adapter 合并与切换
- expected_output: 手写 LoRA 层（PyTorch 100 行以内）+ 在 mini-GPT 上验证效果；再用 PEFT 库在 0.5B 模型上完成同任务
- notes: "手写一遍再上工具"是全书核心方法

### Chapter 7：QLoRA 与显存优化——在消费级硬件微调 7B 模型

- goal: 理解量化原理（NF4）；用 QLoRA 在 7B 模型上完成真实微调；掌握租用云 GPU 的完整流程
- reader_problem: 想微调大模型但只有 8GB 显存/Mac；不知道 QLoRA 为什么能把显存需求降一个数量级
- key_concepts: 量化（Quantization）、NF4、双重量化、QLoRA、梯度检查点（Gradient Checkpointing）、显存估算方法
- expected_output: 一份在 vGPU-32GB 上跑通的 7B QLoRA 完整脚本（含 AutoDL 租用/SSH/传数据/跑训练/取回模型的实操流程）；附 Colab 免费版方案；附 Mac 本地 0.5B 对照版
- notes: 三档硬件方案：Mac 0.5B（本地）→ Colab 免费 → AutoDL 4080/4090

### Chapter 8：偏好对齐 DPO——让模型学会"选更好的答案"

- goal: 理解 SFT 之后为什么还需要对齐；掌握 DPO 的原理与实现；跑通一次小型 DPO 训练
- reader_problem: 不理解 RLHF/PPO/DPO 的关系；没见过 preference pair 数据；想给模型"调性格"但无从下手
- key_concepts: RLHF、奖励模型（Reward Model）、PPO（概念性介绍）、DPO（Direct Preference Optimization）、chosen/rejected 对、参考模型（Reference Model）、KL 约束
- expected_output: 一个小型偏好数据集 + TRL `DPOTrainer` 完整脚本 + 对齐前后回答质量对比
- notes: PPO 只做概念讲解（画清脉络），实操全部用 DPO（简单、稳定、单模型）

## Part 3：评估与落地

### Chapter 9：数据工程——微调成败的一半在数据

- goal: 掌握指令数据的构造、清洗、去重、质量筛选方法；理解"数据质量 > 数据数量"
- reader_problem: 想微调自己领域的模型但不知道数据从哪来、怎么清洗、多少条才够
- key_concepts: 数据来源（人工/自构造/公开集/模型合成）、自指令（Self-Instruct）、去重（精确/模糊）、质量过滤、格式校验、数据配比
- expected_output: 一套可复用的数据清洗脚本（去重 + 过滤 + 格式校验）+ 一份"构造 1000 条领域指令数据"的操作流程
- notes: 强调用 LLM 合成数据的方法，读者可用 API 或本地模型做

### Chapter 10：评估——怎么知道你的模型变好了

- goal: 建立微调项目的评估体系：自动指标（loss/PPL）、benchmark、LLM-as-Judge、人工评估
- reader_problem: 微调完只看 loss 下降，不知道实际效果好坏；不知道该选哪些 benchmark、怎么避免过拟合评估集
- key_concepts: Perplexity、benchmark（MMLU/CMMLU 概览）、LLM-as-Judge、评估集污染、消融对比（baseline 对照）、人工评估量表
- expected_output: 一个评估脚本包（PPL 计算 + LLM-as-Judge 打分模板）+ 一份评估 checklist
- notes: 评估集要和训练集隔离的原则贯穿全书

### Chapter 11：多模态——视觉语言模型 VLM 原理与微调

- goal: 理解 VLM 的结构（视觉编码器 + 投影层 + LLM）；跑通一次小型 VLM 微调（图文指令）
- reader_problem: 不理解图像怎么变成模型能处理的输入；想让模型看图说话但不知道从哪学起
- key_concepts: 视觉编码器（ViT/CLIP）、投影层（Projector/Adapter）、图文对数据、图像 token、VLM 微调策略（冻结 vs 训练投影层）
- expected_output: 一个小型图文微调实验（如 LLaVA 风格小模型或小型开源 VLM + LoRA）+ 看图问答 demo
- notes: 硬件双轨：本地小模型 / GPU 版

### Chapter 12：部署与推理优化——把模型用起来

- goal: 掌握模型导出、量化推理（GGUF/llama.cpp）、高效推理服务（vLLM）、快速搭一个可分享的 Gradio demo
- reader_problem: 训练完的模型只是一堆 checkpoint，不知道怎么变成能用的服务；被各种量化格式（GGUF/GPTQ/AWQ）搞晕
- key_concepts: checkpoint 导出与合并、GGUF 格式、llama.cpp（Mac 原生支持）、vLLM、KV Cache、吞吐 vs 延迟、Gradio demo
- expected_output: 完整的"训练产物 → 量化 → 本地推理 → Web demo"链路脚本，Mac 和 GPU 双路径
- notes: 全书收束章：用第 5–8 章微调的模型做部署，形成完整闭环
