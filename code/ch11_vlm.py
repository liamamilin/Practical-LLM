# ch11_vlm.py
# 环境要求: pip install torch transformers peft pillow torchvision
#           (首次运行下载 SmolVLM-256M-Instruct 约 1GB)
# 硬件: Mac(MPS) 8GB 内存即可 / GPU; 纯 CPU 可跑但慢
# 预计运行: Mac(MPS) 约 50-60 分钟 (MPS 上 VLM 训练较慢, GPU 上约 5 分钟)

import random
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model

torch.manual_seed(42)
random.seed(42)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"设备: {device}")

model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
proc = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, dtype=torch.bfloat16).to(device)
model.config.use_cache = False

# ---------- 第 1 步: 生成合成图文数据集 ----------
# 图 = 彩色圆 + 数字; 问答教模型用固定格式回答

COLORS = ["red", "green", "blue", "orange", "purple", "brown"]


def make_image(color, number):
    img = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([64, 64, 192, 192], fill=color)
    d.text((112, 108), str(number), fill="white")
    return img


def make_sample(color, number):
    img = make_image(color, number)
    q = random.choice([
        f"What number is written in the image?",
        f"What color is the circle in the image?",
    ])
    if "number" in q:
        a = f"The answer is: {number}"
    else:
        a = f"The answer is: {color}"
    return {"image": img, "q": q, "a": a}


all_colors = [(c, n) for c in COLORS for n in range(10)]
random.shuffle(all_colors)
train_pairs = [make_sample(c, n) for c, n in all_colors[:36]]
test_pairs = [make_sample(c, n) for c, n in all_colors[36:40]]
print(f"数据: 训练 {len(train_pairs)} 条, 测试 {len(test_pairs)} 条 (图完全没见过)")


def to_inputs(sample, with_answer):
    """with_answer=True: 训练用 (prompt+答案); False: 推理用"""
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": sample["image"]},
        {"type": "text", "text": sample["q"]}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    if not with_answer:
        return proc(text=[prompt], images=[sample["image"]],
                    return_tensors="pt")
    full = prompt + sample["a"] + "<end_of_utterance>"
    return proc(text=[full], images=[sample["image"]], return_tensors="pt")


# ---------- 第 2 步: 微调前基线 ----------

@torch.no_grad()
def ask(m, sample, max_new_tokens=25):
    inputs = to_inputs(sample, with_answer=False).to(device)
    out = m.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # 只取 Assistant: 之后的部分 (decode 全文含对话模板)
    return proc.batch_decode(out, skip_special_tokens=True)[0].split(
        "Assistant:")[-1].strip()


print("=" * 50)
print("微调前 (基线回答):")
for s in test_pairs[:3]:
    q_short = "数字" if "number" in s["q"] else "颜色"
    print(f"  [{q_short}] {ask(model, s)!r}")


# ---------- 第 3 步: LoRA 微调 (只挂语言模型部分) ----------

def make_labels(sample):
    """labels = 输入副本, prompt 部分 (含图像 token) 置 -100"""
    full = to_inputs(sample, with_answer=True)
    prompt_only = to_inputs(sample, with_answer=False)
    p_len = prompt_only["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :p_len] = -100
    full["labels"] = labels
    return full


lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],   # 只挂语言模型的注意力投影
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=1e-4)

model.train()
for step in range(40):
    batch_samples = random.sample(train_pairs, 2)
    feats = [make_labels(s) for s in batch_samples]
    maxlen = max(f["input_ids"].shape[1] for f in feats)
    input_ids = torch.full((len(feats), maxlen), proc.tokenizer.pad_token_id,
                           dtype=torch.long)
    labels = torch.full((len(feats), maxlen), -100, dtype=torch.long)
    attn = torch.zeros((len(feats), maxlen), dtype=torch.long)
    pixel_rows = []
    for i, f in enumerate(feats):
        L = f["input_ids"].shape[1]
        input_ids[i, :L] = f["input_ids"][0]
        labels[i, :L] = f["labels"][0]
        attn[i, :L] = f["attention_mask"][0]
        pixel_rows.append(f["pixel_values"][0])
    pixel_values = torch.stack(pixel_rows).to(device)
    batch = {"input_ids": input_ids.to(device), "labels": labels.to(device),
             "attention_mask": attn.to(device), "pixel_values": pixel_values}
    out = model(**batch)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    optimizer.zero_grad()
    if step % 10 == 0:
        print(f"step {step:2d} | loss {out.loss.item():.4f}")
        if device == "mps":
            torch.mps.empty_cache()
model.save_pretrained("smolvlm_lora_adapter")
print("适配器已保存: smolvlm_lora_adapter/")

# ---------- 第 4 步: 微调后验证 ----------

model.eval()
print("=" * 50)
print("微调后 (没见过的图, 应输出 'The answer is: ...'):")
ok_fmt, ok_all = 0, 0
for s in test_pairs:
    q_short = "数字" if "number" in s["q"] else "颜色"
    ans = ask(model, s)
    gold = s["a"].replace("The answer is: ", "")
    fmt = ans.startswith("The answer is:")
    hit = gold in ans
    ok_fmt += fmt
    ok_all += (fmt and hit)
    print(f"  [{q_short}] {ans!r}  期望内容 {gold!r} "
          f"格式{'✓' if fmt else '✗'} 内容{'✓' if hit else '✗'}")
print(f"格式正确率: {ok_fmt}/{len(test_pairs)}, "
      f"格式+内容双达标: {ok_all}/{len(test_pairs)}")
