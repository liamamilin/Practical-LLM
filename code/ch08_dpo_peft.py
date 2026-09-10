# ch08_dpo_peft.py
# 环境要求: pip install torch transformers peft trl accelerate datasets
#           (首次运行下载 Qwen2.5-0.5B-Instruct 约 1GB, 复用前面章节的缓存)
# 硬件: Mac(MPS) 16GB 内存 / GPU; 纯 CPU 可跑但很慢
# 预计运行: Mac(MPS) 约 3-5 分钟

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import DPOConfig, DPOTrainer

torch.manual_seed(42)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"设备: {device}")

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
model.config.use_cache = False


# ---------- 第 1 步: 偏好数据 ----------
# 同一个问题, 两种回答: chosen = "【答案】X" 格式, rejected = 普通陈述句
# 训练目标: 让模型学会"偏好"这种格式 —— 不靠指令, 靠偏好信号

capitals = {"中国": "北京", "日本": "东京", "法国": "巴黎", "德国": "柏林",
            "英国": "伦敦", "韩国": "首尔", "意大利": "罗马", "西班牙": "马德里",
            "澳大利亚": "堪培拉", "巴西": "巴西利亚", "俄罗斯": "莫斯科"}
pairs = []
for c, cap in capitals.items():
    q = f"{c}的首都是哪里"
    pairs.append({
        "prompt": [{"role": "user", "content": q}],
        "chosen": [{"role": "assistant", "content": f"【答案】{cap}"}],
        "rejected": [{"role": "assistant", "content": f"{c}的首都是{cap}。"}],
    })
for a in range(1, 16):
    for b in range(1, 16):
        q = f"{a}+{b}等于多少"
        pairs.append({
            "prompt": [{"role": "user", "content": q}],
            "chosen": [{"role": "assistant", "content": f"【答案】{a+b}"}],
            "rejected": [{"role": "assistant", "content": f"{a} + {b} = {a+b}"}],
        })

# 评估用: 模型从未见过的问题 (不在训练集里)
eval_qs = [("埃及的首都是哪里", "开罗", "埃及的首都是开罗。"),
           ("印度的首都是哪里", "新德里", "印度的首都是新德里。"),
           ("17+25等于多少", "42", "17 + 25 = 42"),
           ("1+1等于多少", "2", "1 + 1 = 2")]

train_ds = Dataset.from_list(pairs)
print(f"偏好对: {len(train_ds)} 条 (chosen=【答案】格式, rejected=普通格式)")


# ---------- 第 2 步: TRL DPOTrainer ----------
# 参考模型 = 未加 LoRA 的底座 (ref_model=None 时 TRL 自动处理)

peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj"],
                         task_type="CAUSAL_LM")

dpo_args = DPOConfig(
    output_dir="dpo_out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,      # 等效 batch 8
    max_steps=100,
    learning_rate=5e-5,
    beta=0.5,                           # 偏大: 约束策略贴近参考模型, 防崩坏
    logging_steps=25,
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_args,
    train_dataset=train_ds,
    processing_class=tok,
    peft_config=peft_config,
)
trainer.train()
trainer.save_model("qwen_dpo_adapter")
print("DPO 适配器已保存: qwen_dpo_adapter/")


# ---------- 第 3 步: 评估 —— 隐式奖励边际 ----------
# DPO 优化的是"偏好排序": 对同一问题的两个候选答案,
# 策略模型与参考模型的 logprob 差 (乘 beta) 就是隐式奖励.
# 边际从 0 (与参考一致) 变成正的越大, 说明偏好越强烈地倒向 chosen 格式.

def load(policy):
    base = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    base.config.use_cache = True
    return (PeftModel.from_pretrained(base, "qwen_dpo_adapter")
            .to(device).eval() if policy == "dpo"
            else base.to(device).eval())


@torch.no_grad()
def answer_logprob(m, q, answer_text):
    """答案部分的 logprob 总和 (不含 prompt, 含结尾符)"""
    msgs = [{"role": "user", "content": q}]
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
    full = tok.apply_chat_template(
        msgs + [{"role": "assistant", "content": answer_text}],
        tokenize=False, add_generation_prompt=False)
    p_ids = tok(prompt)["input_ids"]
    f_ids = tok(full)["input_ids"]
    x = torch.tensor([f_ids], device=m.device)
    logits = m(x).logits
    lp = F.log_softmax(logits[0], dim=-1)
    total = 0.0
    for i in range(len(p_ids) - 1, len(f_ids) - 1):
        total += lp[i, f_ids[i + 1]].item()
    return total


ref = load("ref")
dpo = load("dpo")
BETA = 0.5

print("=" * 52)
print("没见过的题上的隐式奖励边际 (正 = 偏好【答案】格式):")
for q, good, bad in eval_qs:
    r_g = answer_logprob(ref, q, f"【答案】{good}")
    r_b = answer_logprob(ref, q, bad)
    d_g = answer_logprob(dpo, q, f"【答案】{good}")
    d_b = answer_logprob(dpo, q, bad)
    margin = BETA * ((d_g - r_g) - (d_b - r_b))
    print(f"  Q: {q:14s} 边际 = {margin:+.2f}")
