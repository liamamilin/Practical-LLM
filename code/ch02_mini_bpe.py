# ch02_mini_bpe.py
# 环境要求: 纯 Python, 无第三方依赖
# 硬件: 任意 (CPU 即可)
# 预计运行: < 5 秒

# ---------- 第 1 部分: 手写 BPE ----------

def get_pair_counts(ids):
    """统计相邻 token 对的出现次数"""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    """把序列中所有出现 pair 的位置合并成 new_id"""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class MiniBPE:
    def __init__(self):
        self.merges = {}  # (id, id) -> 新 id, 编码规则
        self.vocab = {}   # 新 id -> bytes, 解码表

    def train(self, text, vocab_size, verbose=False):
        """从文本训练 BPE: 反复合并出现次数最多的相邻 token 对"""
        ids = list(text.encode("utf-8"))  # 起点: 256 种字节
        num_merges = vocab_size - 256
        for step in range(num_merges):
            counts = get_pair_counts(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = 256 + step
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            if verbose:
                print(f"merge {step:3d}: {pair} -> {new_id} "
                      f"(出现 {counts[pair]} 次)")
        # 构建解码表: 每个 id 对应的实际字节串
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

    def encode(self, text):
        """文本 -> token 序列: 按训练时的优先级逐个应用合并规则"""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            # 找当前序列中"合并优先级最高"(最早学到的) 的 token 对
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        """token 序列 -> 文本"""
        return b"".join(self.vocab[i] for i in ids).decode(
            "utf-8", errors="replace")


# ---------- 第 2 部分: 训练语料 (小型中英混合文本) ----------

corpus = (
    "大语言模型通过预测下一个词来学习语言。"
    "大语言模型需要大量文本数据。语言模型是人工智能的核心技术。"
    "人工智能正在改变世界,语言模型是人工智能的重要分支。"
    "我们训练语言模型,然后微调语言模型,最后部署语言模型。"
    "The large language model learns language by prediction. "
    "Language models need a lot of text data. "
    "We train the model, then fine-tune the model, then deploy the model."
) * 8  # 重复 8 次让高频模式更明显

# ---------- 第 3 部分: 训练 + 编码 + 解码 ----------

if __name__ == "__main__":
    bpe = MiniBPE()
    bpe.train(corpus, vocab_size=280, verbose=True)

    print("=" * 50)
    test = "大语言模型学习语言。"
    ids = bpe.encode(test)
    recovered = bpe.decode(ids)
    print(f"原文: {test!r}")
    print(f"token 数: {len(ids)} (UTF-8 字节数: {len(test.encode('utf-8'))})")
    print(f"token 序列: {ids}")
    print(f"解码还原: {recovered!r}")
    print(f"还原一致: {recovered == test}")
    print(f"压缩率: {len(ids) / len(test.encode('utf-8')):.2%}")

    # 解码每个 token, 看它"是什么"
    print("-" * 50)
    print("每个 token 对应的字节串:")
    for i in ids:
        token_bytes = bpe.vocab[i]
        try:
            pretty = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            pretty = repr(token_bytes)
        print(f"  {i:3d} -> {pretty!r}")
