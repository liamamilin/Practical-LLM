# ch09_data_engineering.py
# 环境要求: 纯 Python (可选演示需 transformers, 见文末)
# 硬件: 任意
# 预计运行: < 5 秒

import hashlib
import re
import random

random.seed(42)

# ---------- 第 1 步: 造一个"脏"数据集 (模拟真实采集结果) ----------

raw = []

def add(q, a, src):
    raw.append({"q": q, "a": a, "src": src})

# 1a. 规矩样本
for i, (c, cap) in enumerate([("中国", "北京"), ("日本", "东京"), ("法国", "巴黎"),
                              ("德国", "柏林"), ("英国", "伦敦"), ("韩国", "首尔"),
                              ("意大利", "罗马"), ("西班牙", "马德里")]):
    add(f"{c}的首都是哪里", cap, "手写")

# 1b. 完全重复 (常见于多源爬取)
add("中国的首都是哪里", "北京", "爬虫A")
add("中国的首都是哪里", "北京", "爬虫B")

# 1c. 近似重复 (同一个问题换标点/换问法/错别字)
add("中国的 首都 是哪里？", "北京", "爬虫A")
add("请问中国首都叫什么", "北京", "爬虫B")
add("法国的首都是哪里呀", "巴黎", "爬虫C")   # 口语化变体, 序列相似度 > 0.9

# 1d. 空答案 / 太短
add("澳大利亚的首都是哪里", "", "爬虫A")
add("巴西的首都是哪里", "。", "爬虫A")

# 1e. 答案过长 (跑题、夹带私货)
add("埃及的首都是哪里",
    "开罗。开罗是埃及的首都也是最大城市，位于尼罗河畔。埃及是一个历史悠久的"
    "国家，有金字塔等著名古迹。顺便说一下，埃及的棉花很有名，每年出口大量"
    "长绒棉。尼罗河全长六千六百多公里，是世界上最长的河流之一。", "爬虫A")

# 1f. 中英混杂严重 (语言不纯)
add("俄罗斯的首都是哪里", "Moscow 莫斯科 Moscow 莫斯科 Moscow 莫斯科", "爬虫B")

# 1g. 高重复度答案 (复读机式生成)
add("乌克兰的首都是哪里", "基辅基辅基辅基辅基辅基辅基辅基辅", "模型生成")
add("匈牙利的首都是哪里", "布达佩斯布达佩斯布达佩斯布达佩斯", "模型生成")

# 1h. 格式错误
add("", "布加勒斯特", "爬虫A")            # 空问题
add("葡萄牙的首都是哪里", None, "爬虫B")   # 缺字段

print(f"原始数据: {len(raw)} 条")
for i, r in enumerate(raw):
    print(f"  [{i}] q={r['q']!r:24} a={(r['a'] or '')[:20]!r}")


# ---------- 第 2 步: 清洗管线 (四个阶段) ----------

def clean_pipeline(items, fuzzy_threshold=0.5):
    report = {}

    # 阶段 1: 格式校验 —— 结构完整性
    kept = []
    for it in items:
        if not it.get("q") or not it.get("a"):
            continue
        if len(it["a"]) < 1:
            continue
        kept.append(it)
    report["格式校验"] = len(items) - len(kept)

    # 阶段 2: 语言过滤 —— 中文占比
    def zh_ratio(text):
        zh = len(re.findall(r"[\u4e00-\u9fff]", text))
        return zh / max(len(text), 1)

    stage_in = len(kept)
    kept = [it for it in kept
            if zh_ratio(it["a"]) > 0.3 and zh_ratio(it["q"]) > 0.3]
    report["语言过滤"] = stage_in - len(kept)

    # 阶段 3: 精确去重 —— 规范化后哈希
    def normalize(text):
        text = re.sub(r"\s+", "", text)          # 去空白
        text = re.sub(r"[，。？?！!、]", "", text)  # 去标点
        return text.lower()

    def sig(it):
        return hashlib.md5(normalize(it["q"]).encode()).hexdigest()

    stage_in = len(kept)
    seen, uniq = set(), []
    for it in kept:
        s = sig(it)
        if s not in seen:
            seen.add(s)
            uniq.append(it)
    kept = uniq
    report["精确去重"] = stage_in - len(kept)

    # 阶段 4: 模糊去重 —— 序列相似度 (difflib), 阈值按数据分布调
    import difflib

    def sim(a, b):
        return difflib.SequenceMatcher(
            None, normalize(a), normalize(b)).ratio()

    stage_in = len(kept)
    survivors = []
    keep_flags = []
    for i, it in enumerate(kept):
        is_dup = any(
            sim(it["q"], kept[j]["q"]) > 0.9
            for j, flag in enumerate(keep_flags) if flag)
        keep_flags.append(not is_dup)
        if not is_dup:
            survivors.append(it)
    kept = survivors
    report["模糊去重"] = stage_in - len(kept)

    # 阶段 5: 质量规则 —— 答案长度 / 重复度
    def repetition_ratio(text):
        chars = list(text)
        if len(chars) < 4:
            return 0.0
        return 1 - len(set(chars)) / len(chars)

    def quality_ok(it):
        if not (2 <= len(it["a"]) <= 60):        # 答案长度合理区间
            return False
        if repetition_ratio(it["a"]) > 0.6:      # 复读机答案
            return False
        return True

    stage_in = len(kept)
    kept = [it for it in kept if quality_ok(it)]
    report["质量规则"] = stage_in - len(kept)

    return kept, report


cleaned, report = clean_pipeline(raw)
print("=" * 50)
print("清洗报告 (每阶段删除条数):")
for stage, n in report.items():
    print(f"  {stage}: -{n}")
print(f"最终: {len(raw)} -> {len(cleaned)} 条")
print("-" * 50)
print("幸存样本:")
for it in cleaned:
    print(f"  q={it['q']!r} a={it['a']!r}")


# ---------- 第 3 步: 模板化数据扩增 (构造是常态) ----------

verbs = ["背诵", "默写", "朗读", "解释"]
templates = ["帮我用一句话介绍{city}", "{city}是哪个国家的首都", "随便说说{city}"]
augmented = []
for it in cleaned[:3]:
    city = it["a"]
    country = it["q"].replace("的首都是哪里", "")
    for t in templates:
        augmented.append({"q": t.format(city=city),
                          "a": f"{city}是{country}的首都。"})
print("=" * 50)
print(f"模板扩增示例 (3 个问题 × 3 个模板 = {len(augmented)} 条):")
for it in augmented[:4]:
    print(f"  Q: {it['q']}")
    print(f"  A: {it['a']}")


# ---------- 第 4 步 (可选): 用本地 LLM 合成数据 ----------

# 真实流程中, 模板扩产之后通常用 LLM 做开放式合成 (Self-Instruct 思路):
#   种子问题 -> LLM 生成新问题 -> LLM 生成答案 -> 同样的清洗管线过滤
# 下面用第 1 章的 Qwen-0.5B 演示 (需要 transformers, 生成 3 条示例):

USE_LLM = False   # 改成 True 体验 (Mac MPS 约 2 分钟)

if USE_LLM:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).to(device).eval()

    seeds = ["写出3个以'为什么'开头的关于地理的问题, 每行一个, 不要编号"]
    msgs = [{"role": "user", "content": seeds[0]}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)
    out = model.generate(**ids, max_new_tokens=100, do_sample=True,
                         temperature=0.9, top_p=0.9)
    synthesized = tok.decode(out[0][ids["input_ids"].shape[1]:],
                             skip_special_tokens=True)
    print("=" * 50)
    print("LLM 合成的新问题 (之后还要逐条生成答案并过清洗管线):")
    print(synthesized)
