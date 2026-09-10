# Writing State

## Book Metadata

- title: 动手做大模型：从零训练到微调实战
- subtitle: 从零训练到微调实战
- author: 米霖（已确认）
- book_type: technical / tutorial
- target_reader: 有编程基础的工程师，不要求深度学习背景
- total_parts: 3
- total_chapters: 12

## Chapter Manifest

| NN | Part | Chapter Title | File | Status | Notes |
|---|---|---|---|---|---|
| 01 | 1 | 大模型全景 | chapter-01-llm-overview.qmd | completed | 已在 Mac(MPS) 实机验证；本地 venv: /tmp/ch01venv (torch 2.14/transformers 5.16) |
| 02 | 1 | Tokenizer：手写 BPE | chapter-02-tokenizer-bpe.qmd | completed | 两个脚本均在本地验证（纯 Python + HF 对照） |
| 03 | 1 | Attention 与 Transformer：手写 mini-GPT | chapter-03-mini-gpt.qmd | completed | 已在 Mac(MPS) 验证：0.14M 参数，初始 loss 5.77 ≈ ln(256) |
| 04 | 1 | 预训练 | chapter-04-pretraining.qmd | completed | 已在 Mac(MPS) 验证：1000 步 4.4s，loss 5.74→0.10，三档温度采样正常；曾修复双重错位 bug |
| 05 | 2 | 指令微调 SFT | chapter-05-sft.qmd | completed | 已验证：微调前后对比 + 未见题全错但格式正确（掩码 SFT） |
| 06 | 2 | PEFT：手写 LoRA | chapter-06-lora-peft.qmd | completed | 两个脚本均验证：手写版(r=8,16.4K,10.68%) + PEFT 版(0.5B,0.149%,未见题泛化) |
| 07 | 2 | QLoRA 与显存优化 | chapter-07-qlora.qmd | completed | **GPU 实跑成功**：7B 峰值显存 8.9GB，可训练 0.53%，loss 4.38→0.001，未见题文言文泛化；显存账表已更新为实测 |
| 08 | 2 | 偏好对齐 DPO | chapter-08-dpo.qmd | completed | 手写版(按对采样bug修复+β调参) + TRL版(隐式奖励边际评估,未见题+5.6~11.4) 均验证 |
| 09 | 3 | 数据工程 | chapter-09-data-engineering.qmd | completed | 清洗管线验证：21→9 条，五阶段各司其职；含 LLM 合成可选演示 |
| 10 | 3 | 评估 | chapter-10-evaluation.qmd | completed | PPL(1.19 vs 2031) + LLM-as-Judge(4/5, 位置偏差实锤) 验证通过 |
| 11 | 3 | 多模态 VLM | chapter-11-vlm.qmd | completed | SmolVLM-256M 微调验证：内容 4/4，格式 2/4（40步）；MPS 上约 78s/步 |
| 12 | 3 | 部署与推理优化 | chapter-12-deployment.qmd | completed | llama.cpp 218 tok/s + merge(部署prompt坑已记录) + Gradio HTTP 200 验证 |

## Terminology

见 book-principles.md 第 8 节，以其为准。

## Chapter Summaries

| Chapter | Summary | Key Terms | Open Threads |
|---|---|---|---|

## Open Issues

- [ ] Quarto 渲染验证：本机未安装 quarto（brew cask 需 sudo 密码），用户本地安装后运行 `quarto render` 检查构建（终检已通过：YAML 合法、围栏配对、frontmatter、术语、章号互引、产物文件对应、无 TODO）
- [ ] AutoDL 实验完成，建议用户关机；SSH 密码已在对话中暴露，建议控制台改密
- [ ] qwen7b_qlora_adapter（7B 文言文适配器）留在 AutoDL 网盘未取回（用户已确认不需要）；复现步骤完整记录在 docs/gpu-runbook.md 实测版

## GPU 待用物料（已就绪）

- `gpu_bundle.tar.gz`（12MB，**已同步至最终版代码**，含 ch07 修复）：15 个章节脚本 + mini-GPT 四个 checkpoint + 三个 LoRA 适配器（qwen_merged 1.9GB 不打包，由 base+adapter 重建）
- `docs/gpu-runbook.md`：完整 GPU 实验流程（已执行完毕）
- `code/qlora_run.log`：第 7 章 GPU 实跑完整日志（已归档）
- GPU 实跑产出（在服务器 /root，可随时取回）：qwen7b_qlora_adapter/（7B 文言文适配器）

## Progress Log

- 2026-09-09: bootstrap 完成（_quarto.yml、index.qmd、references.qmd、references.bib）；content.md 12 章蓝图完成；book-principles.md 完成；AutoDL SSH 免密配置完成；训练栈安装完成（torch 2.5.1+cu124, transformers 5.16.1, peft 0.20.0, trl 1.12.0, datasets 5.0.1, accelerate 1.14.0, bitsandbytes 0.50.2, sentencepiece 0.2.2）
- 2026-09-09: 第 1 章写完并在 Mac(MPS) 验证通过（Qwen2.5-0.5B-Instruct 推理 + top5 概率），示例输出已替换为真实运行结果
- 2026-09-10: 第 2-8 章完成并逐一本地验证（手写 BPE / mini-GPT / 预训练 4.4s / SFT 掩码 / 手写 LoRA + PEFT 0.5B / DPO 手写+TRL）；修复双重错位、LoRA 设备搬移、DPO 配对采样三个教学级 bug（均写入对应章节）
- 2026-09-10: 第 9-12 章完成并验证（数据清洗 21→9 / PPL 1.19vs2031 + Judge 4/5 / SmolVLM 微调 内容4/4 格式2/4 / llama.cpp 218tok/s + merge + Gradio）
- 2026-09-10: 前言完成；_quarto.yml 12 章顺序就位；audit 通过。待办：GPU 批量实验（ch07 实跑）+ 用户本地安装 Quarto 渲染
- 2026-09-10: **GPU 批量实验完成**：ch04 GPU 对照（15.3s vs MPS 4.4s，小模型 GPU 无优势——已写入第 4 章）；ch07 QLoRA 实跑成功（7B 峰值显存 8.9GB，文言文风格泛化到未见题——真实输出已替换占位文本，显存账表更新为实测值）；过程修复 ch07 pairs tuple/dict bug。服务器上留有 qwen7b_qlora_adapter 可取回。**全书 12 章全部完成且全部实机验证**
- 2026-09-10: **概念—操作能力原则落地**：重新核对 skill 第 9 条要求后，book-principles.md 新增第 8 节（Concept–Operation 框架 + 掌握层级链 + 正文/代码分工定位）；12 章的"本章产物"全部补写「概念—操作增量」（Concept / Operation / Judgment 三行），全书围栏与结构复查通过
