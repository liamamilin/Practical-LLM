# ch06_lora_peft.py
# 环境要求: pip install torch transformers peft accelerate
#           (首次运行下载 Qwen2.5-0.5B-Instruct 约 1GB, 复用第 1 章缓存)
# 硬件: Mac(MPS) 8GB 内存即可 / GPU 更快; 纯 CPU 可跑但慢
# 预计运行: Mac(MPS) 约 3-5 分钟

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

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
model = model.to(device)
model.config.use_cache = False          # 训练时关闭 KV 缓存


# ---------- 第 1 步: 构造"只输出答案"风格的指令数据 ----------

capitals = {"中国": "北京", "日本": "东京", "法国": "巴黎", "德国": "柏林",
            "英国": "伦敦", "韩国": "首尔", "意大利": "罗马", "西班牙": "马德里"}
math_pairs = [(f"{a}+{b}等于多少", str(a + b))
              for a in range(1, 16) for b in range(1, 16)]

qa_train = ([{"q": f"{c}的首都是哪里", "a": cap} for c, cap in capitals.items()]
            + [{"q": q, "a": a} for q, a in math_pairs])
qa_test = [  # 未参与训练的题
    {"q": "埃及的首都是哪里", "a": "开罗"},
    {"q": "12+13等于多少", "a": "25"},
]

SYSTEM = "你是一个乐于助人的中文助手。"
# 训练目标: 让模型养成"【答案】X"的固定应答格式 (只靠示例, 不靠指令)


def to_ids(q, a=None):
    """用 chat template 编码; 推理时 a=None, 加生成提示符"""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": q}]
    if a is None:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    else:
        msgs.append({"role": "assistant", "content": f"【答案】{a}"})
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=False)
    return tok(text)["input_ids"]


def collate(pairs):
    """pad 成等长 batch, 并用 -100 掩掉答案之外的所有位置"""
    encs = [to_ids(p["q"], p["a"]) for p in pairs]
    maxlen = max(len(e) for e in encs)
    x = torch.full((len(encs), maxlen), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((len(encs), maxlen), -100, dtype=torch.long)
    attn = torch.zeros((len(encs), maxlen), dtype=torch.long)
    for i, e in enumerate(encs):
        x[i, :len(e)] = torch.tensor(e)
        attn[i, :len(e)] = 1
        labels[i, :len(e)] = torch.tensor(e)   # 先全部设为标签
    # 掩掉答案之前的部分: 对每条重新计算 prompt 长度
    for i, p in enumerate(pairs):
        prompt_len = len(to_ids(p["q"]))       # 含生成提示符的 prompt 长度
        labels[i, :prompt_len] = -100
    return x.to(device), attn.to(device), labels.to(device)


# ---------- 第 2 步: 挂 LoRA (HuggingFace PEFT) ----------

lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=1e-4)


# ---------- 第 3 步: 训练 ----------

model.train()
for step in range(40):
    idx = torch.randperm(len(qa_train))[:8]
    batch = [qa_train[i] for i in idx]
    x, attn, labels = collate(batch)
    out = model(input_ids=x, attention_mask=attn, labels=labels)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    optimizer.zero_grad()
    if step % 10 == 0:
        print(f"step {step:2d} | loss {out.loss.item():.4f}")
model.save_pretrained("qwen_lora_adapter")   # 只保存 LoRA 增量
print("LoRA 适配器已保存: qwen_lora_adapter/")


# ---------- 第 4 步: 前后对比 ----------

@torch.no_grad()
def answer(q, max_new_tokens=60):
    ids = torch.tensor(to_ids(q), device=device).unsqueeze(0)
    out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens,
                         do_sample=False, temperature=None,
                         top_p=None, top_k=None)
    new = out[0][ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip()


model.eval()
print("=" * 50)
for p in qa_train[:1] + qa_test:
    print(f"Q: {p['q']}  (期望: {p['a']})")
    print(f"  模型: {answer(p['q'])!r}")
