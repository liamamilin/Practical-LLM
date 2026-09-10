# ch12_deploy.py
# 环境要求: pip install torch transformers peft gradio llama-cpp-python
#           (llama-cpp-python 在 Mac 上有预编译 wheel; Linux 需 cmake)
#           需要第 6 章的 qwen_lora_adapter/ 和 ch06/ch12 下载过的模型
# 硬件: Mac 16GB 内存即可 (合并模型需 ~4GB); llama.cpp 部分 4GB 即可
# 预计运行: Mac 约 3-5 分钟

import glob
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

torch.manual_seed(42)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"设备: {device}")

# ============================================================
# 第 1 部分: llama.cpp —— 4-bit/8-bit 量化的本地推理
# (GGUF 模型已提前下载, 下载命令见本章正文)

print("=" * 50)
print("第 1 部分: llama.cpp 量化推理")
from llama_cpp import Llama

ggufs = glob.glob(
    "**/qwen2.5-0.5b-instruct-q8_0.gguf", root_dir="/private/var/folders",
    recursive=True)
# 通用写法: 指向你自己的 GGUF 文件路径
# llm = Llama(model_path="/path/to/qwen2.5-0.5b-instruct-q8_0.gguf", n_ctx=2048)

if ggufs:
    gguf_path = "/private/var/folders/" + ggufs[0]
    llm = Llama(model_path=gguf_path, n_ctx=2048, verbose=False)
    msgs = [{"role": "user", "content": "用一句话解释什么是量化。"}]
    t0 = time.time()
    out = llm.create_chat_completion(messages=msgs, max_tokens=80,
                                     temperature=0.7)
    dt = time.time() - t0
    n = out["usage"]["completion_tokens"]
    print(f"  回答: {out['choices'][0]['message']['content'].strip()!r}")
    print(f"  速度: {n / dt:.0f} tokens/s | 模型占用 ~0.5GB (Q8 量化)")

# ============================================================
# 第 2 部分: 合并 LoRA 产物 —— 得到可独立发布的完整模型

print("=" * 50)
print("第 2 部分: 合并 LoRA 适配器")
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
base = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
merged = PeftModel.from_pretrained(base, "qwen_lora_adapter")
merged = merged.merge_and_unload()          # 把 LoRA 增量加回原权重
merged.save_pretrained("qwen_merged")
tok.save_pretrained("qwen_merged")
print(f"  合并完成 -> qwen_merged/ (约 2GB, 可独立分发)")


@torch.no_grad()
def chat(message, max_new_tokens=50):
    # 注意: prompt 必须与训练时一致 (含 system 消息) —— 部署最常见的事故
    # 就是部署侧 prompt 与训练侧不一致, 导致微调行为"消失"
    text = tok.apply_chat_template(
        [{"role": "system",
          "content": "你是一个乐于助人的中文助手。"},
         {"role": "user", "content": message}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)
    out = merged.to(device).generate(
        **ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


print(f"  合并后行为检查 (应保留 '【答案】' 格式):")
print(f"    Q: 法国的首都是哪里 -> {chat('法国的首都是哪里')!r}")
merged = merged.to("cpu")
if device == "mps":
    torch.mps.empty_cache()


# ============================================================
# 第 3 部分: Gradio —— 给模型一个能分享的界面

print("=" * 50)
print("第 3 部分: Gradio Web Demo")

import gradio as gr


def respond(message, history):
    return chat(message, max_new_tokens=100)


demo = gr.ChatInterface(
    fn=respond,
    title="我的微调模型 (第 6 章 LoRA 合并版)",
    description="Qwen2.5-0.5B + LoRA (教它用【答案】格式回答)。本地运行。",
)

if __name__ == "__main__":
    print("启动 Web 界面: http://127.0.0.1:7860")
    demo.launch()          # 本地调试; 公网分享用 share=True (需网络)
