import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"使用设备: {device}")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
model = model.to(device)
model.eval()

messages = [{"role": "user", "content": "用一句话解释什么是大语言模型。"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("-" * 40)
print("送入模型的原始文本:")
print(text)

inputs = tokenizer(text, return_tensors="pt").to(device)
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9)

new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
print("-" * 40)
print("模型回答:")
print(answer)

next_step = model(**inputs)
probs = torch.softmax(next_step.logits[0, -1], dim=-1)
top5 = torch.topk(probs, 5)
print("-" * 40)
print("模型眼中, 对话的下一个 token 概率 top5:")
for idx, p in zip(top5.indices, top5.values):
    print(f"  {tokenizer.decode([idx])!r}: {p:.4f}")
