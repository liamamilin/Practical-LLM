# ch06_lora_manual.py
# 环境要求: pip install torch; 需要同目录下的 ch03_mini_gpt.py
# 硬件: CPU / Mac(MPS) / GPU 均可
# 预计运行: Mac(MPS) 约 10 秒

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from ch03_mini_gpt import MiniGPT, CONFIG, device

torch.manual_seed(42)

# ---------- 第 1 步: 手写 LoRA 层 ----------

class LoRALinear(nn.Module):
    """在冻结的原线性层旁边加一条低秩旁路:
       输出 = W0 @ x + (alpha/r) * B @ A @ x
       A: 降维 (d -> r), B: 升维 (r -> d), B 初始化为 0 (训练开始时旁路无扰动)"""

    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base                       # 原层, 稍后冻结
        d_out, d_in = base.weight.shape
        self.lora_A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.lora_A.T @ self.lora_B.T)


def apply_lora(model, target_names=("c_attn",), r=4, alpha=8):
    """把模型里名字在 target_names 中的 Linear 替换成 LoRALinear"""
    replaced = 0
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if child_name in target_names and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r, alpha))
                replaced += 1
    return replaced


# ---------- 第 2 步: 复用第 5 章的指令数据集 (内联以保持单文件可运行) ----------

capitals = {
    "中国": "北京", "日本": "东京", "法国": "巴黎", "德国": "柏林",
    "英国": "伦敦", "韩国": "首尔", "意大利": "罗马", "西班牙": "马德里",
    "加拿大": "渥太华", "澳大利亚": "堪培拉", "巴西": "巴西利亚",
    "埃及": "开罗", "俄罗斯": "莫斯科", "印度": "新德里",
}
math_pairs = [(f"{a}+{b}等于多少", str(a + b))
              for a in range(1, 12) for b in range(1, 12)]
qa_train = ([{"q": f"{c}的首都是哪里", "a": cap} for c, cap in capitals.items()]
            + [{"q": q, "a": a} for q, a in math_pairs])


def render(q, a=None):
    return f"Q: {q}\nA:" if a is None else f"Q: {q}\nA: {a}\n"


def sft_loss(logits, labels, mask):
    V = logits.size(-1)
    raw = F.cross_entropy(logits[:, :-1].reshape(-1, V),
                          labels[:, 1:].reshape(-1), reduction="none")
    m = mask[:, 1:].reshape(-1).float()
    return (raw * m).sum() / m.sum()


def make_batch(pairs):
    texts = [render(p["q"], p["a"]) for p in pairs]
    enc = [list(t.encode("utf-8")) for t in texts]
    maxlen = max(len(e) for e in enc)
    x = torch.zeros((len(enc), maxlen), dtype=torch.long)
    mask = torch.zeros((len(enc), maxlen), dtype=torch.bool)
    for i, (e, t) in enumerate(zip(enc, texts)):
        x[i, :len(e)] = torch.tensor(e)
        a_start = len(t[:t.index("A: ") + 3].encode("utf-8"))
        mask[i, a_start - 1:len(e)] = True
    return x.to(device), mask.to(device)


# ---------- 第 3 步: 只训 LoRA 旁路 ----------

model = MiniGPT(CONFIG).to(device)
model.load_state_dict(torch.load("mini_gpt_ckpt.pt"))   # 预训练底座

n = apply_lora(model, target_names=("c_attn", "c_fc", "c_proj"),
               r=8, alpha=16)
model = model.to(device)        # 新建的 LoRA 参数也要搬到设备上
print(f"替换了 {n} 个线性层")

for p in model.parameters():
    p.requires_grad = False                              # 冻结全部原权重
for m in model.modules():
    if isinstance(m, LoRALinear):
        m.lora_A.requires_grad = True
        m.lora_B.requires_grad = True

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数: {total / 1e6:.2f}M | 可训练: {trainable / 1e3:.1f}K "
      f"({trainable / total:.2%})")

x, mask = make_batch(qa_train)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-3)

model.train()
t0 = time.time()
for step in range(1200):
    idx = torch.randperm(x.size(0))[:16]
    logits, _ = model(x[idx])
    loss = sft_loss(logits, x[idx], mask[idx])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    optimizer.zero_grad()
    if step % 300 == 0:
        print(f"step {step:3d} | sft loss {loss.item():.3f} | {time.time()-t0:.0f}s")
torch.save({k: v for k, v in model.state_dict().items()
            if "lora" in k}, "mini_gpt_lora.pt")
print(f"LoRA 增量已保存: mini_gpt_lora.pt "
      f"({sum(v.numel() for v in torch.load('mini_gpt_lora.pt').values()) / 1e3:.1f}K 参数)")


# ---------- 第 4 步: 验证效果 ----------

@torch.no_grad()
def generate(model, prompt, max_new_tokens=40, temperature=0.3, top_k=10):
    idx = torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long,
                       device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -CONFIG["block_size"]:])
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")
        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        if idx[0, -1].item() == ord("\n"):
            break
    return bytes(idx[0].tolist()).decode("utf-8", errors="replace")


model.eval()
print("=" * 50)
for p in ["法国的首都是哪里", "3+5等于多少", "2+9等于多少"]:
    prompt = render(p)
    print(f"{prompt!r}")
    print(f"  -> {generate(model, prompt)!r}")
