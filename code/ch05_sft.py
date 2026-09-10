# ch05_sft.py
# 环境要求: pip install torch; 需要同目录下的 ch03_mini_gpt.py
# 硬件: CPU / Mac(MPS) / GPU 均可 (模型约 200 万参数)
# 预计运行: Mac(MPS) 约 1 分钟, 纯 CPU 约 5 分钟

import time
import torch
import torch.nn.functional as F

from ch03_mini_gpt import MiniGPT, CONFIG, device

torch.manual_seed(42)

# ---------- 第 1 步: 构造小型指令数据集 ----------

# 格式: "Q: {问题}\nA: {答案}\n"
# 训练目标: 只对答案部分 (A: 之后) 计算 loss —— 模型不需要学怎么"提问"

capitals = {
    "中国": "北京", "日本": "东京", "法国": "巴黎", "德国": "柏林",
    "英国": "伦敦", "韩国": "首尔", "意大利": "罗马", "西班牙": "马德里",
    "加拿大": "渥太华", "澳大利亚": "堪培拉", "巴西": "巴西利亚",
    "埃及": "开罗", "俄罗斯": "莫斯科", "印度": "新德里",
}

math_pairs = [(f"{a}+{b}等于多少", str(a + b))
              for a in range(1, 12) for b in range(1, 12)]

qa_all = [{"q": f"{c}的首都是哪里", "a": cap} for c, cap in capitals.items()]
qa_all += [{"q": q, "a": a} for q, a in math_pairs]

# 留出 2 个国家 + 2 道算术做"没见过的题"
qa_test = [qa_all.pop(0), qa_all.pop(5)]          # 中国→北京, 日本→东京
qa_test += [qa_all.pop(20), qa_all.pop(21)]       # 两道算术题

qa_train = qa_all
print(f"指令数据: 训练 {len(qa_train)} 条, 测试(未见过) {len(qa_test)} 条")


def render(q, a=None):
    """把问答对渲染成训练/推理用的文本格式"""
    if a is None:
        return f"Q: {q}\nA:"
    return f"Q: {q}\nA: {a}\n"


# ---------- 第 2 步: 带掩码的 SFT 损失 ----------

# 沿用 ch04 的模型配置 (2 层 / 64 维 / 窗口 64), 权重可以直接继承
model = MiniGPT(CONFIG).to(device)
try:                                    # 从第 4 章的预训练底座出发
    model.load_state_dict(torch.load("mini_gpt_ckpt.pt"))
    print("已加载 ch04 预训练权重作为底座")
except FileNotFoundError:
    print("未找到预训练权重, 从随机初始化开始")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)


def sft_loss(logits, labels, mask):
    """只对 mask=True 的位置计算交叉熵 (答案部分)"""
    V = logits.size(-1)
    shift_logits = logits[:, :-1].reshape(-1, V)
    shift_labels = labels[:, 1:].reshape(-1)
    shift_mask = mask[:, 1:].reshape(-1).float()
    raw = F.cross_entropy(shift_logits, shift_labels, reduction="none")
    return (raw * shift_mask).sum() / shift_mask.sum()


def make_batch(pairs, bs=16):
    """把问答对 pad 成等长 batch, 并生成答案区掩码"""
    texts = [render(p["q"], p["a"]) for p in pairs]
    enc = [list(t.encode("utf-8")) for t in texts]
    maxlen = max(len(e) for e in enc)
    pad_id = 0
    x = torch.full((len(enc), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((len(enc), maxlen), dtype=torch.bool)
    for i, (e, t) in enumerate(zip(enc, texts)):
        x[i, :len(e)] = torch.tensor(e)
        # 答案起点: "A: " 之后; 用字符串定位, 字节级对齐
        a_start = t.index("A: ") + 3
        a_start = len(t[:a_start].encode("utf-8"))
        mask[i, a_start - 1:len(e)] = True     # 从答案首字节到最后
    return x.to(device), mask.to(device)


# ---------- 第 3 步: SFT 训练循环 ----------

x, mask = make_batch(qa_train)
print(f"batch 形状: {tuple(x.shape)}, 答案区占比 {mask.float().mean():.1%}")

model.train()
t0 = time.time()
for step in range(600):
    idx = torch.randperm(x.size(0))[:16]       # 小数据集: 每 step 抽 16 条
    logits, _ = model(x[idx])
    loss = sft_loss(logits, x[idx], mask[idx])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    if step % 100 == 0:
        print(f"step {step:3d} | sft loss {loss.item():.3f} | "
              f"{time.time()-t0:.0f}s")
torch.save(model.state_dict(), "mini_gpt_sft.pt")


# ---------- 第 4 步: 对比"微调前 vs 微调后" ----------

@torch.no_grad()
def generate(model, prompt, max_new_tokens=60, temperature=0.3, top_k=10):
    idx = torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long,
                       device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -CONFIG["block_size"]:])
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        if idx[0, -1].item() == ord("\n"):      # 生成到换行即结束
            break
    return bytes(idx[0].tolist()).decode("utf-8", errors="replace")


base = MiniGPT(CONFIG).to(device)
base.load_state_dict(torch.load("mini_gpt_ckpt.pt"))
base.eval()
model.eval()

print("=" * 50)
for label, m in [("微调前 (只会续写)", base), ("微调后 (学会回答)", model)]:
    for p in [{"q": "法国的首都是哪里"}, {"q": "3+5等于多少"}]:
        prompt = render(p["q"])
        out = generate(m, prompt)
        print(f"[{label}] {prompt!r}")
        print(f"  -> {out!r}")
print("=" * 50)
print("没见过的题 (考察是否记住格式、能否泛化):")
for p in qa_test[:4]:
    prompt = render(p["q"])
    out = generate(model, prompt)
    truth = p["a"]
    print(f"  Q: {p['q']}  正确答案: {truth}")
    print(f"  模型: {out!r}")
