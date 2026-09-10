# ch10_evaluation.py
# 环境要求: pip install torch transformers
#           需要 ch03_mini_gpt.py / ch04 与 ch05 的 checkpoint / 第 1 章下载过的 Qwen-0.5B
# 硬件: Mac(MPS) / GPU; 纯 CPU 可跑但慢
# 预计运行: Mac(MPS) 约 2 分钟

import math
import re
import random
import torch
import torch.nn.functional as F

from ch03_mini_gpt import MiniGPT, CONFIG, device

torch.manual_seed(42)

# ============================================================
# 第 1 部分: Perplexity —— 语言模型自己的"读起来有多顺"
# ============================================================

model = MiniGPT(CONFIG).to(device)
model.load_state_dict(torch.load("mini_gpt_ckpt.pt"))
model.eval()


@torch.no_grad()
def perplexity(model, text, block=64, max_windows=20):
    """PPL = exp(平均交叉熵)。模型在没见过的文本上逐窗口计算。"""
    data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    losses = []
    step = max(1, (len(data) - block - 1) // max_windows)
    for i in range(0, len(data) - block - 1, step):
        x = data[i:i + block].unsqueeze(0).to(device)
        _, loss = model(x, targets=x)
        losses.append(loss.item())
    return math.exp(sum(losses) / len(losses))


in_domain = ("大语言模型通过预测下一个词来学习语言。"
             "语言模型是人工智能的核心技术之一。"
             "我们训练语言模型,然后微调语言模型,最后部署语言模型。"
             "预测下一个词看起来简单,却让模型学会了语法和知识。" * 3)
out_domain = ("床前明月光,疑是地上霜。举头望明月,低头思故乡。"
              "两个黄鹂鸣翠柳,一行白鹭上青天。窗含西岭千秋雪,"
              "门泊东吴万里船。")
print("第 1 部分: Perplexity")
print(f"  训练语料同风格文本 : PPL = {perplexity(model, in_domain):.2f}")
print(f"  域外文本 (古诗)   : PPL = {perplexity(model, out_domain):.2f}")

# ============================================================
# 第 2 部分: LLM-as-Judge —— 用强模型当裁判
# ============================================================

from transformers import AutoModelForCausalLM, AutoTokenizer

judge_name = "Qwen/Qwen2.5-0.5B-Instruct"
jt = AutoTokenizer.from_pretrained(judge_name)
judge = AutoModelForCausalLM.from_pretrained(
    judge_name, dtype=torch.float32).to(device).eval()

sft = MiniGPT(CONFIG).to(device)
sft.load_state_dict(torch.load("mini_gpt_sft.pt"))
sft.eval()
base = MiniGPT(CONFIG).to(device)
base.load_state_dict(torch.load("mini_gpt_ckpt.pt"))
base.eval()


@torch.no_grad()
def generate_mini(model, prompt, max_new_tokens=40, temperature=0.3, top_k=10):
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


QUESTIONS = ["法国的首都是哪里", "德国的首都是哪里", "韩国的首都是哪里",
             "3+5等于多少", "6+7等于多少"]

JUDGE_PROMPT = """你是一个评审。以下是同一个问题的两个回答。

问题: {q}

回答A: {a}
回答B: {b}

哪个回答更准确、更切题? 只回复一个字母: A 或 B"""


@torch.no_grad()
def judge_pick(q, ans_a, ans_b, labels=("A", "B")):
    """让裁判模型选择更好的回答。
    用 A/B 两个 token 的概率比较, 比自由生成更稳定 (且可解释)。"""
    text = jt.apply_chat_template(
        [{"role": "user",
          "content": JUDGE_PROMPT.format(q=q, a=ans_a, b=ans_b)}],
        tokenize=False, add_generation_prompt=True)
    ids = jt(text, return_tensors="pt").to(device)
    logits = judge(**ids).logits[0, -1]
    id_a = jt(labels[0], add_special_tokens=False)["input_ids"][0]
    id_b = jt(labels[1], add_special_tokens=False)["input_ids"][0]
    pair = torch.tensor([logits[id_a].item(), logits[id_b].item()])
    pa, pb = F.softmax(pair, dim=0).tolist()
    return labels[0] if pa > pb else labels[1], pa, pb


print("=" * 50)
print("第 2 部分: LLM-as-Judge (裁判 = Qwen-0.5B, 选手 = mini-GPT 两个版本)")
print(f"{'问题':<14} {'基座':<6} {'SFT后':<6} 裁判判给")

def after_answer(text):
    """截取 'A:' 之后的部分 —— 裁判只应看到回答本身"""
    return text.split("A:", 1)[-1].strip()


judge_score = {"base": 0, "sft": 0}
for q in QUESTIONS:
    m = re.search(r"(\d+)\+(\d+)", q)
    if m:
        gold = str(int(m.group(1)) + int(m.group(2)))
    else:
        gold = {"法国": "巴黎", "德国": "柏林", "韩国": "首尔"}.get(q[:2])
    ans_base = after_answer(generate_mini(base, f"Q: {q}\nA:"))
    ans_sft = after_answer(generate_mini(sft, f"Q: {q}\nA:"))
    # 位置随机化: 防止裁判偏爱固定位置 (位置偏差)
    if random.random() < 0.5:
        verdict, pa, pb = judge_pick(q, ans_base, ans_sft)
        winner = {"A": "base", "B": "sft"}[verdict]
    else:
        verdict, pb, pa = judge_pick(q, ans_sft, ans_base)
        winner = {"A": "sft", "B": "base"}[verdict]
    judge_score[winner] += 1
    print(f"{q:<12} 基座:{ans_base[:8]!r:<14} SFT:{ans_sft[:6]!r:<10} "
          f"-> {verdict} (p={max(pa, pb):.2f}) [{winner}]")

print("-" * 50)
n = len(QUESTIONS)
print(f"裁判结果: 基座 {judge_score['base']} 票 | SFT {judge_score['sft']} 票")
print("(金标准: SFT 模型每题都应胜出 —— 它才是能回答问题的那个)")
