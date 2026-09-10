# ch08_dpo_manual.py
# 环境要求: pip install torch; 需要同目录下的 ch03_mini_gpt.py
# 硬件: CPU / Mac(MPS) / GPU 均可
# 预计运行: Mac(MPS) 约 10 秒
# 前置: 运行过 ch05_sft.py (需要 mini_gpt_sft.pt)

import time
import torch
import torch.nn.functional as F

from ch03_mini_gpt import MiniGPT, CONFIG, device

torch.manual_seed(42)
BETA = 0.5   # DPO 温度: 越大越贴紧参考模型, 越小越激进

# ---------- 第 1 步: 构造偏好对 ----------

capitals = {
    "中国": "北京", "日本": "东京", "法国": "巴黎", "德国": "柏林",
    "英国": "伦敦", "韩国": "首尔", "意大利": "罗马", "西班牙": "马德里",
    "加拿大": "渥太华", "澳大利亚": "堪培拉", "巴西": "巴西利亚",
    "埃及": "开罗", "俄罗斯": "莫斯科", "印度": "新德里",
}
cities = list(capitals.values())

prefs = []   # (question, chosen_answer, rejected_answer)
for i, (c, cap) in enumerate(capitals.items()):
    wrong = cities[(i + 7) % len(cities)]     # 拿另一个首都当"坏答案"
    prefs.append((f"{c}的首都是哪里", cap, wrong))
for a in range(1, 12):
    for b in range(1, 12):
        prefs.append((f"{a}+{b}等于多少", str(a + b), str(a + b + 1)))

# 留 4 对做评估 (模型没见过的问题也留一对)
test_prefs = [prefs.pop(0), prefs.pop(5), prefs.pop(-1), prefs.pop(-2)]
print(f"偏好对: 训练 {len(prefs)} 对, 评估 {len(test_prefs)} 对")


def render(q, a=None):
    return f"Q: {q}\nA:" if a is None else f"Q: {q}\nA: {a}\n"


def encode_pair(q, chosen, rejected):
    """把一对答案编码成两条等格式序列 + 答案区掩码"""
    out = []
    for a in (chosen, rejected):
        t = render(q, a)
        e = list(t.encode("utf-8"))
        mask_start = len(t[:t.index("A: ") + 3].encode("utf-8"))
        m = [False] * len(e)
        for j in range(mask_start - 1, len(e)):
            m[j] = True
        out.append((e, m))
    return out


# ---------- 第 2 步: 序列级 logprob (DPO 的原料) ----------

def seq_logprobs(model, x, mask):
    """每个样本在答案区的 logprob 总和, 返回 shape (B,)"""
    logits, _ = model(x)
    logprobs = F.log_softmax(logits[:, :-1], dim=-1)
    labels = x[:, 1:]
    tok_lp = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    m = mask[:, 1:].float()
    return (tok_lp * m).sum(dim=1)


def stack(encs):
    """把变长 (字节序列, 掩码) 列表 pad 成等长张量"""
    maxlen = max(len(e) for e, _ in encs)
    xs = torch.zeros((len(encs), maxlen), dtype=torch.long)
    ms = torch.zeros((len(encs), maxlen), dtype=torch.bool)
    for i, (e, m) in enumerate(encs):
        xs[i, :len(e)] = torch.tensor(e)
        ms[i, :len(e)] = torch.tensor(m)
    return xs.to(device), ms.to(device)


@torch.no_grad()
def pair_accuracy(model, prefs_list):
    """chosen 的 logprob 高于 rejected 的比例"""
    wins = 0
    for q, c, r in prefs_list:
        xs, ms = stack(encode_pair(q, c, r))
        lp = seq_logprobs(model, xs, ms)
        wins += (lp[0] > lp[1]).item()
    return wins / len(prefs_list)


# ---------- 第 3 步: 策略模型 (从 SFT 出发) + 冻结的参考模型 ----------

policy = MiniGPT(CONFIG).to(device)
policy.load_state_dict(torch.load("mini_gpt_sft.pt"))    # 第 5 章 SFT 模型
ref = MiniGPT(CONFIG).to(device)
ref.load_state_dict(torch.load("mini_gpt_sft.pt"))       # 同起点, 之后冻结
for p in ref.parameters():
    p.requires_grad = False
ref.eval()

print(f"训练前 pair accuracy: {pair_accuracy(policy, test_prefs):.0%}")

optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-5)

# 把训练对编码成张量
xs, ms = [], []
for q, c, r in prefs:
    for e, m in encode_pair(q, c, r):
        xs.append(e)
        ms.append(m)
maxlen = max(len(e) for e in xs)
x_all = torch.zeros((len(xs), maxlen), dtype=torch.long)
m_all = torch.zeros((len(xs), maxlen), dtype=torch.bool)
for i, (e, m) in enumerate(zip(xs, ms)):
    x_all[i, :len(e)] = torch.tensor(e)
    m_all[i, :len(e)] = torch.tensor(m)
x_all, m_all = x_all.to(device), m_all.to(device)


# ---------- 第 4 步: DPO 训练循环 ----------

def dpo_loss(pol_pc, pol_pr, ref_pc, ref_pr):
    """DPO: 最大化 (策略-参考) 的 chosen-rejected 边际"""
    chosen_margins = (pol_pc - ref_pc) - (pol_pr - ref_pr)
    return -F.logsigmoid(BETA * chosen_margins).mean()


policy.train()
t0 = time.time()
for step in range(400):
    # 按对采样: 先抽 16 个对的编号, 再取出每对的 chosen(偶)/rejected(奇) 两行
    pidx = torch.randperm(x_all.size(0) // 2)[:16]
    idx = torch.stack([pidx * 2, pidx * 2 + 1], dim=1).flatten()
    xb, mb = x_all[idx], m_all[idx]
    pol_lp = seq_logprobs(policy, xb, mb)
    pol_c, pol_r = pol_lp[0::2], pol_lp[1::2]    # 交错排列: 偶=chosen
    with torch.no_grad():
        ref_lp = seq_logprobs(ref, xb, mb)
    ref_c, ref_r = ref_lp[0::2], ref_lp[1::2]
    loss = dpo_loss(pol_c, pol_r, ref_c, ref_r)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    if step % 100 == 0:
        with torch.no_grad():
            margin = ((pol_c - ref_c) - (pol_r - ref_r)).mean()
        print(f"step {step:3d} | dpo loss {loss.item():.4f} "
              f"| 平均边际 {margin.item():+.3f} | {time.time()-t0:.0f}s")

torch.save(policy.state_dict(), "mini_gpt_dpo.pt")
print(f"训练后 pair accuracy: {pair_accuracy(policy, test_prefs):.0%}")


# ---------- 第 5 步: 看行为变化 ----------

@torch.no_grad()
def rank_answers(model, q, answers):
    """模型对多个候选答案的偏好排序"""
    encs = []
    for a in answers:
        t = render(q, a)
        e = list(t.encode("utf-8"))
        m_start = len(t[:t.index("A: ") + 3].encode("utf-8"))
        m = [False] * len(e)
        for j in range(m_start - 1, len(e)):
            m[j] = True
        encs.append((e, m))
    xs, ms = stack(encs)
    lp = seq_logprobs(model, xs, ms)
    order = sorted(zip(answers, lp.tolist()), key=lambda t: -t[1])
    return order


print("=" * 50)
for q, answers in [("法国的首都是哪里", ["巴黎", "伦敦", "罗马"]),
                   ("7+8等于多少", ["15", "16", "70"])]:
    print(f"Q: {q}")
    for label, m in [("微调后策略模型", policy), ("冻结参考模型", ref)]:
        ranked = rank_answers(m, q, answers)
        print(f"  [{label}] " + " > ".join(
            f"{a}({lp:+.2f})" for a, lp in ranked))
