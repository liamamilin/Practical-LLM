# ch03_mini_gpt.py
# 环境要求: pip install torch
# 硬件: CPU / Mac(MPS) / GPU 均可, 参数量约 100 万
# 预计运行: < 5 秒 (只做前向传播, 不训练; 训练在第 4 章)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 自动检测设备: 有 NVIDIA GPU 用 cuda, Mac 用 mps, 否则 cpu
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# ---------- 模型配置 (刻意做小, CPU 上秒级) ----------

CONFIG = {
    "vocab_size": 256,     # 词表: 256 种字节 (字符级简化版, 复用第 2 章思路)
    "block_size": 64,      # 上下文窗口: 一次最多看 64 个 token
    "n_layer": 2,          # Transformer Block 数
    "n_head": 2,           # 注意力头数
    "n_embd": 64,          # 隐层维度 (必须能被 n_head 整除)
}


# ---------- 块 1: 因果自注意力 ----------

class CausalSelfAttention(nn.Module):
    """多头自注意力 + 因果掩码: 每个 token 只能看到它左边的 token"""

    def __init__(self, config):
        super().__init__()
        assert config["n_embd"] % config["n_head"] == 0
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        # Q/K/V 三个投影合一个大矩阵 (c_attn), 输出维度 3 * n_embd
        self.c_attn = nn.Linear(config["n_embd"], 3 * config["n_embd"])
        self.c_proj = nn.Linear(config["n_embd"], config["n_embd"])

    def forward(self, x):
        B, T, C = x.shape                     # batch, 序列长, 隐层维度
        qkv = self.c_attn(x)                  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # 拆出多头: (B, T, C) -> (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # 注意力打分: Q 和 K 的点积 / sqrt(head_dim)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)

        # 因果掩码: 位置 i 不允许看到位置 > i
        mask = torch.tril(torch.ones(T, T, device=x.device)) == 0
        att = att.masked_fill(mask, float("-inf"))

        att = F.softmax(att, dim=-1)          # 归一化成注意力权重
        y = att @ v                           # 加权求和: (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)                 # 输出投影


# ---------- 块 2: 前馈网络 ----------

class MLP(nn.Module):
    """两层全连接: 4 倍升维 -> 非线性 -> 降维回来"""

    def __init__(self, config):
        super().__init__()
        n_embd = config["n_embd"]
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


# ---------- 块 3: Transformer Block ----------

class Block(nn.Module):
    """注意力 + FFN, 各配一条残差连接, 前面各放一个 LayerNorm"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config["n_embd"])
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config["n_embd"])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # 残差: 让深层网络能稳定训练
        x = x + self.mlp(self.ln2(x))
        return x


# ---------- 块 4: 组装成完整 GPT ----------

class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config["block_size"]
        # token embedding + 位置 embedding: 模型只知道"是什么", 还要知道"在哪"
        self.tok_emb = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.pos_emb = nn.Embedding(config["block_size"], config["n_embd"])
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config["n_layer"])])
        self.ln_f = nn.LayerNorm(config["n_embd"])
        # 输出头: 隐层向量 -> 词表上每个 token 的打分 (logits)
        self.head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"序列长 {T} 超过窗口 {self.block_size}"

        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)   # (B, T) -> (B, T, C)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.head(x)                       # (B, T, vocab_size)
        if targets is None:
            return logits, None
        # 交叉熵: 每个位置预测"下一个 token", 所以错开一位对齐
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            targets[:, 1:].reshape(-1))
        return logits, loss


# ---------- 验证: 随机初始化的模型应该猜对多少? ----------

if __name__ == "__main__":
    torch.manual_seed(42)
    model = MiniGPT(CONFIG).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"设备: {device}")
    print(f"参数量: {n_params / 1e6:.2f}M (Qwen-0.5B 约是它的 3500 倍)")

    # 造一个 batch: 4 条序列, 每条 32 个随机 token
    x = torch.randint(0, CONFIG["vocab_size"], (4, 32), device=device)
    logits, loss = model(x, targets=x)

    print(f"logits 形状: {tuple(logits.shape)}  (batch, 序列长, 词表)")
    print(f"初始 loss: {loss:.4f}")
    print(f"理论随机 loss: ln(256) = {math.log(CONFIG['vocab_size']):.4f}")

    # 生成: 从纯噪声开始接龙 40 个 token (随机初始化, 输出也是噪声)
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(40):
            idx_cond = idx[:, -CONFIG["block_size"]:]   # 裁剪到窗口内
            logits, _ = model(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)  # 只取最后一个位置
            next_id = torch.multinomial(probs, 1)        # 按概率采样
            idx = torch.cat([idx, next_id], dim=1)
    print(f"随机初始化模型生成: {''.join(chr(i % 128) for i in idx[0].tolist())!r}")
