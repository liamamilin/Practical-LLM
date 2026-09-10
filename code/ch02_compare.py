# ch02_compare.py
# 环境要求: pip install transformers  (tokenizers 是它的依赖, 会一起装上)
# 硬件: 任意
# 预计运行: < 10 秒 (复用第 1 章已下载的 Qwen tokenizer)

from tokenizers import Tokenizer, models, trainers
from transformers import AutoTokenizer

# ---------- 第 1 部分: 用 HuggingFace 的 BPE 训练同一个语料 ----------

corpus = (
    "大语言模型通过预测下一个词来学习语言。"
    "大语言模型需要大量文本数据。语言模型是人工智能的核心技术。"
    "人工智能正在改变世界,语言模型是人工智能的重要分支。"
    "我们训练语言模型,然后微调语言模型,最后部署语言模型。"
    "The large language model learns language by prediction. "
    "Language models need a lot of text data. "
    "We train the model, then fine-tune the model, then deploy the model."
) * 8

hf_tok = Tokenizer(models.BPE())  # 默认字节级 BPE, 和我们手写的同一原理
trainer = trainers.BpeTrainer(vocab_size=280, show_progress=False)
hf_tok.train_from_iterator([corpus], trainer=trainer)

test = "大语言模型学习语言。"
enc = hf_tok.encode(test)
print("HuggingFace BPE (同语料训练, 词表 280):")
print(f"  token 数: {len(enc.tokens)} -> {enc.tokens}")
print(f"  解码还原: {hf_tok.decode(enc.ids)!r}")

# ---------- 第 2 部分: 真实的 Qwen tokenizer ----------

print("=" * 50)
qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
print(f"Qwen 词表大小: {qwen.vocab_size} (我们的 mini-BPE 只有 280)")

tests = [
    "大语言模型学习语言。",
    "The weather is nice today.",
    "我今天心情很好,想去公园散步。",
]
for t in tests:
    n = len(qwen.encode(t))
    nbytes = len(t.encode("utf-8"))
    nchar = len(t)
    print(f"  {t!r}")
    print(f"    字符 {nchar} | 字节 {nbytes} | token {n}")

# 特殊 token: 不是从文本里"切"出来的, 是词表里预留的固定编号
specials = ["<|im_start|>", "<|im_end|>"]
for s in specials:
    print(f"  特殊 token {s!r} -> id {qwen.convert_tokens_to_ids(s)}")

# 同一句话, 字符 vs token 的差距 (中文按字切, 英文按词根切)
print("=" * 50)
demo = qwen.tokenize("大语言模型学习语言。")
print(f"Qwen 切分 '大语言模型学习语言。' 的 token:")
for t in demo:
    print(f"  {t!r}")
