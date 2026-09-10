# ch04_pretrain.py
# 环境要求: pip install torch; 需要同目录下的 ch03_mini_gpt.py
# 硬件: CPU / Mac(MPS) / GPU 均可
# 预计运行: Mac(MPS) 约 1 分钟, 纯 CPU 约 5-10 分钟

import math
import time
import torch
import torch.nn.functional as F

from ch03_mini_gpt import MiniGPT, CONFIG, device

torch.manual_seed(42)

# ---------- 第 1 步: 训练语料 (字节级, 复用第 3 章的 256 词表) ----------

corpus_text = (
    "大语言模型通过预测下一个词来学习语言。"
    "语言模型是人工智能的核心技术之一。"
    "人工智能正在改变世界,语言模型是人工智能的重要分支。"
    "我们训练语言模型,然后微调语言模型,最后部署语言模型。"
    "预测下一个词看起来简单,却让模型学会了语法和知识。"
    "训练模型需要数据、算力和耐心。"
    "微调让模型学会听指令,对齐让模型回答得更好。"
    "大语言模型是人工智能的重要方向。"
    "小模型可以在笔记本上训练,大模型需要成千上万块显卡。"
    "学习语言模型的最好方法是亲手训练一个小模型。"
) * 40  # 约 40KB 字节; 重复次数人为调高, 让小模型能学到稳定的模式

data = torch.tensor(list(corpus_text.encode("utf-8")), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]   # 评估集必须与训练集隔离
print(f"语料: {len(data)} 字节 (约 {len(data)//3} 个汉字)")

CONFIG["block_size"] = 64
B, T = 12, 64                                # batch, 序列长


def get_batch(split):
    """从字节流里随机裁剪 batch 条长度为 T 的序列"""
    stream = train_data if split == "train" else val_data
    ix = torch.randint(len(stream) - T - 1, (B,))
    x = torch.stack([stream[i:i + T] for i in ix])
    return x.to(device)
    # 注意: 不需要手动构造目标序列, MiniGPT.forward 内部会自动
    # "错开一位" (第 k 个位置的输出对齐第 k+1 个输入作为标签)


# ---------- 第 2 步: 优化器与学习率调度 ----------

model = MiniGPT(CONFIG).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

max_steps = 1000
warmup_steps = 50


def get_lr(step):
    """前 50 步线性升温, 之后余弦退火到 1e-4 (生产训练的标准做法)"""
    if step < warmup_steps:
        return 1e-3 * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + math.cos(math.pi * progress))


# ---------- 第 3 步: 训练循环 ----------

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(10):
            x = get_batch(split)
            _, loss = model(x, targets=x)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


print("=" * 50)
t0 = time.time()
history = []
for step in range(max_steps):
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    x = get_batch("train")
    _, loss = model(x, targets=x)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
    optimizer.step()
    optimizer.zero_grad()

    if step % 100 == 0 or step == max_steps - 1:
        losses = estimate_loss()
        history.append((step, losses["train"], losses["val"]))
        print(f"step {step:4d} | lr {lr:.2e} | "
              f"train loss {losses['train']:.3f} | val loss {losses['val']:.3f}")

print(f"训练完成, 耗时 {time.time() - t0:.1f} 秒")
torch.save(model.state_dict(), "mini_gpt_ckpt.pt")
print("checkpoint 已保存: mini_gpt_ckpt.pt")


# ---------- 第 4 步: 采样生成 ----------

@torch.no_grad()
def generate(prompt, max_new_tokens=120, temperature=0.8, top_k=20):
    """温度控制随机性, top-k 只在概率前 k 的 token 中采样"""
    idx = torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long,
                       device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -CONFIG["block_size"]:]      # 裁剪到窗口内
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx[0].tolist()


print("=" * 50)
for temp in [0.3, 0.8, 1.5]:
    out = generate("大语言模型是", temperature=temp)
    text = bytes(out).decode("utf-8", errors="replace")
    print(f"[temperature={temp}]")
    print(f"  {text}")
    print()
