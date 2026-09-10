# GPU 实验手册（实测版）——第 7 章 QLoRA 的完整复现步骤

以下步骤于 2026-09-10 在 AutoDL 实例（RTX 4080 SUPER vGPU / 32GB / CUDA 12.4 镜像 / ¥1.58/时）上**真实执行通过**。全程费用约 ¥2。

## 步骤 0：租用与开机

1. AutoDL 控制台 → GPU 云主机 → 选 RTX 4080/4090（vGPU-32GB 更稳）
2. 镜像：**基础镜像 / PyTorch 2.5.1 / Python 3.12 / CUDA 12.4**
3. 数据盘 50GB 起；**正常有卡模式开机**
4. 控制台复制 SSH 登录指令（形如 `ssh -p 19449 root@connect.westc.seetacloud.com`）

## 步骤 1：SSH 免密（本地只需一次）

```bash
ssh-copy-id -p 19449 root@connect.westc.seetacloud.com
# 验证（应看到 GPU 信息）:
ssh -p 19449 root@connect.westc.seetacloud.com \
    'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
# 预期: NVIDIA GeForce RTX 4080 SUPER, 32760 MiB
```

## 步骤 2：上传代码（本地打包 → scp）

```bash
# 本地: 打包脚本 + checkpoint + 适配器 (不含 1.9GB 的 qwen_merged, 可重建)
tar czf gpu_bundle.tar.gz -C code ch01_hello_llm.py ch02_compare.py \
    ch02_mini_bpe.py ch03_mini_gpt.py ch04_pretrain.py ch05_sft.py \
    ch06_lora_manual.py ch06_lora_peft.py ch07_qlora_gpu.py \
    ch08_dpo_manual.py ch08_dpo_peft.py ch09_data_engineering.py \
    ch10_evaluation.py ch11_vlm.py ch12_deploy.py \
    mini_gpt_ckpt.pt mini_gpt_sft.pt mini_gpt_lora.pt mini_gpt_dpo.pt \
    qwen_lora_adapter qwen_dpo_adapter smolvlm_lora_adapter

scp -P 19449 gpu_bundle.tar.gz root@connect.westc.seetacloud.com:/root/
ssh -p 19449 root@connect.westc.seetacloud.com \
    'cd /root && tar xzf gpu_bundle.tar.gz'   # xattr 警告无害
```

## 步骤 3：环境补装（唯一要小心的坑：版本钉死）

```bash
ssh -p 19449 root@connect.westc.seetacloud.com \
  "source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && \
   pip install -q pillow && \
   pip install -q --no-deps torchvision==0.20.1 && \
   python -c 'import torch, torchvision; \
     assert torch.__version__.startswith(\"2.5.1\"); print(\"OK\")'"
```

**为什么 --no-deps**：直接 `pip install torchvision` 会把 torch 连带升级到不匹配的
CUDA 构建，训练环境直接报废。torch 2.5.1 配 torchvision 0.20.1。

## 步骤 4：GPU 对照验证（可选，书里第 4 章的数据来自这步）

```bash
ssh -p 19449 root@connect.westc.seetacloud.com \
  "source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && \
   export HF_ENDPOINT=https://hf-mirror.com && cd /root && \
   python ch04_pretrain.py"
# 实测: GPU 15.3 秒 (Mac MPS 4.4 秒) —— 小模型 GPU 无优势, 见第 4 章
```

## 步骤 5：QLoRA 主实验（第 7 章）

```bash
ssh -p 19449 root@connect.westc.seetacloud.com \
  "source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && \
   export HF_ENDPOINT=https://hf-mirror.com \
          HF_HOME=/root/autodl-tmp/hf_cache && \
   cd /root && \
   (setsid nohup python -u ch07_qlora_gpu.py > qlora_run.log 2>&1 < /dev/null &)"

# 轮询进度:
ssh -p 19449 root@connect.westc.seetacloud.com \
    'grep -E "trainable|step|峰值|适配器|A:" /root/qlora_run.log | tail -10'
```

预期输出（本书实测）：

```text
trainable params: 40,370,176 || all params: 7,655,986,688 || trainable%: 0.5273
step  0 | loss 4.3806 | 峰值显存 8.8 GB
...
step 50 | loss 0.0010 | 峰值显存 8.9 GB
适配器已保存: qwen7b_qlora_adapter/
Q: 你是谁
  A: '吾乃一语言模型，通晓古今文字。'
Q: 什么是量子计算        # 未参与训练
  A: '量子计算者，以量子比特为单位，利用叠加与纠缠之律，行并行运算之术也。'
```

耗时与费用：模型下载约 12 分钟（15GB，hf-mirror）+ 训练约 3 分钟，共约 ¥1。

## 步骤 6：收尾

```bash
# 取回日志
scp -P 19449 root@connect.westc.seetacloud.com:/root/qlora_run.log .
# 需要模型则取适配器 (~160MB):
scp -P 19449 -r root@connect.westc.seetacloud.com:/root/qwen7b_qlora_adapter .
# 然后控制台关机 (按开机时长计费)
```

## 实测踩坑记录（每一条都真实发生过）

| 坑 | 现象 | 修法 |
|---|---|---|
| **hf-xet 与镜像不兼容** | 下载报 401 Unauthorized（xet 协议走了 hf.co 的 CAS 服务器） | `pip uninstall -y hf_xet`；或设 `HF_HUB_DISABLE_XET=1` |
| **模型缓存撑爆系统盘** | 系统盘仅 30GB，7B 模型缓存 15GB | 设 `HF_HOME=/root/autodl-tmp/hf_cache` 把缓存指到数据盘 |
| **pkill 自杀** | `pkill -f ch07_qlora` 把执行它的 SSH 会话自己也杀了（命令行含同样字符串） | 模式写成 `ch07_[q]lora` 避免自匹配，或按 PID 杀 |
| **tmux 未安装** | 该镜像没有 tmux | 用 `setsid nohup ... < /dev/null &` 替代；或先 `apt install tmux` |
| **HF_ENDPOINT 未设** | 直连 huggingface.co 超时 | 每次导出 `HF_ENDPOINT=https://hf-mirror.com` |
| **pairs 数据结构不一致** | 训练开跑即 `TypeError: tuple indices must be integers` | 数据统一转 dict：`pairs = [{"q": q, "a": a} for q, a in pairs]` |
