# ch07_qlora_gpu.py
# 环境要求: GPU (显存 >= 16GB, 4090/vGPU-32GB 均可)
#           pip install torch transformers peft accelerate bitsandbytes datasets
# 运行方式: 建议在 tmux 中运行 (SSH 断开不中断训练), 见本章正文操作流程
# 预计运行: 7B 模型下载约 15GB; 60 步训练约 10-20 分钟
# 硬件对照: Mac 无法运行 bitsandbytes 4-bit, Mac 方案见正文"替代路线"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = "Qwen/Qwen2.5-7B-Instruct"

# ---------- 第 1 步: 以 NF4 4-bit 加载底座 ----------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # QLoRA 论文的 4-bit 格式
    bnb_4bit_use_double_quant=True,       # 对量化常数再量化, 每参数再省 0.4 bit
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时反量化到 bf16
)

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map={"": 0},                   # 全部放进 0 号卡
)
model.config.use_cache = False
model.gradient_checkpointing_enable()     # 用时间换显存: 不存中间激活
model.enable_input_require_grads()        # 梯度检查点 + LoRA 所需
model = prepare_model_for_kbit_training(model)

# ---------- 第 2 步: 挂 LoRA ----------

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=1e-4)

# ---------- 第 3 步: 数据: 教模型用文言文回答 ----------

pairs = [
    ("你是谁", "吾乃一语言模型，通晓古今文字。"),
    ("你好", "幸会，君有何事垂询？"),
    ("今天天气怎么样", "吾无目，不知风雨，然闻君言矣。"),
    ("你会什么", "吾能属文、对答、译言，请君试之。"),
    ("谢谢", "不必言谢，能助君乃吾之幸。"),
    ("再见", "山高水长，后会有期。"),
    ("什么是人工智能", "人工智能者，机巧之学也，使机器似人而思。"),
    ("什么是大语言模型", "大语言模型者，读万卷书，习言语之律，故能对答如流。"),
    ("你怎么工作的", "吾以文字为食，逐字预测，连缀成句。"),
    ("给我讲个笑话", "一日，程序员谓机曰：除一错。机曰：诺，然增二错。"),
    ("你好呀", "君安，请讲。"),
    ("你叫什么名字", "吾无名，姑且称吾为语言模型。"),
    ("现在几点了", "吾无钟表，不知时辰，君可自观之。"),
    ("帮我写首诗", "秋风起兮白云飞，代码成兮人不归。"),
    ("英语的苹果怎么说", "apple 是也。"),
    ("一加一等于几", "其一加其一，得二。"),
    ("你聪明吗", "不敢称聪，然博闻强记，愿效犬马。"),
    ("我饿了怎么办", "宜速饭，饭毕再来。"),
    ("你会说英语吗", "然，吾亦通英语，君可试之。"),
    ("苹果手机多少钱", "物价无常，吾不知价，君宜问市。"),
    ("什么是机器学习", "机器学习者，使机器自数据中习规律之术也。"),
    ("什么是微调", "微调者，以小数据改模型之习，如教人改口音也。"),
    ("你觉得我聪明吗", "君既问此，必有过人之处。"),
    ("讲个故事", "昔有一生徒，夜读不辍，终成大器。"),
    ("睡觉重要吗", "然，眠足则神清，神清则学敏。"),
]
pairs = [{"q": q, "a": a} for q, a in pairs]   # 统一成 dict, 方便后续按字段访问
SYSTEM = "你是一个说文言文的助手, 所有回答必须用文言文。"


def to_ids(q, a=None):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": q}]
    if a is None:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    else:
        msgs.append({"role": "assistant", "content": a})
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=False)
    return tok(text)["input_ids"]


def collate(p_list):
    encs = [to_ids(p["q"], p["a"]) for p in p_list]
    maxlen = max(len(e) for e in encs)
    x = torch.full((len(encs), maxlen), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((len(encs), maxlen), -100, dtype=torch.long)
    attn = torch.zeros((len(encs), maxlen), dtype=torch.long)
    for i, e in enumerate(encs):
        x[i, :len(e)] = torch.tensor(e)
        attn[i, :len(e)] = 1
        labels[i, :len(e)] = torch.tensor(e)
    for i, p in enumerate(p_list):
        labels[i, :len(to_ids(p["q"]))] = -100
    return x.to("cuda"), attn.to("cuda"), labels.to("cuda")


# ---------- 第 4 步: 训练 ----------

model.train()
for step in range(60):
    idx = torch.randperm(len(pairs))[:4]
    x, attn, labels = collate([pairs[i] for i in idx])
    out = model(input_ids=x, attention_mask=attn, labels=labels)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    optimizer.zero_grad()
    if step % 10 == 0:
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"step {step:2d} | loss {out.loss.item():.4f} "
              f"| 峰值显存 {mem:.1f} GB")
model.save_pretrained("qwen7b_qlora_adapter")
print("适配器已保存: qwen7b_qlora_adapter/")


# ---------- 第 5 步: 测试 (含没见过的题) ----------

model.eval()
test_qs = ["你是谁", "什么是量子计算", "给我一句话介绍你自己"]
with torch.no_grad():
    for q in test_qs:
        ids = torch.tensor(to_ids(q), device="cuda").unsqueeze(0)
        out = model.generate(input_ids=ids, max_new_tokens=80, do_sample=False)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"Q: {q}\n  A: {ans.strip()!r}")
