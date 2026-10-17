# -*- coding: utf-8 -*-
"""数据集密钥清洗规则：

1. 剔除含 ToolBench 污染的样本（assistant 参数里带 toolbench_rapidapi_key 字段，
   且混有真实第三方 RapidAPI key）；
2. 将样本内容中内嵌的高德 key 替换为占位符 REDACTED_AMAP_KEY。

数据文件为 JSON 数组、每项一条样本；本脚本就地修改，修改前请先备份。
用法：python sanitize_dataset.py [文件1 文件2 ...]（不带参数则洗默认清单）
"""
import io
import json
import os
import re
import sys

def _find_project_root(start_dir):
    """向上定位项目根（以 .env.example / rag-system 目录为标记）"""
    d = os.path.abspath(start_dir)
    while True:
        if os.path.exists(os.path.join(d, ".env.example")) or os.path.isdir(os.path.join(d, "rag-system")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.abspath(start_dir)
        d = parent


ROOT = _find_project_root(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "datasets")
AMAP_PLACEHOLDER = "REDACTED_AMAP_KEY"
CONTAMINATION_MARK = "toolbench_rapidapi_key"

DEFAULT_FILES = [
    "merged_test_final.json",
    "converted_test_final/converted_batch_003.json",
    "merged_train_final.json",
    "merged_train_final_multiturn.json",
    "merged_train_final_multiturn_v2.json",
    "converted_train_final/converted_batch_028.json",
]


def load_amap_key():
    env_path = os.path.join(ROOT, ".env")
    if os.path.exists(env_path):
        for line in io.open(env_path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("AMAP_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("AMAP_KEY", "")


def detect_indent(path):
    head = io.open(path, encoding="utf-8").read(4000)
    m = re.match(r"\[\s*\n( +)", head)
    return len(m.group(1)) if m else 2


def clean_file(path, amap_key):
    if not os.path.exists(path):
        print(f"[跳过] {path}: 文件不存在")
        return
    indent = detect_indent(path)
    data = json.load(io.open(path, encoding="utf-8"))
    if not isinstance(data, list):
        print(f"[跳过] {path}: 不是样本数组")
        return
    kept = [s for s in data if CONTAMINATION_MARK not in json.dumps(s, ensure_ascii=False).lower()]
    dropped = len(data) - len(kept)
    text = json.dumps(kept, ensure_ascii=False, indent=indent)
    if amap_key:
        text = text.replace(amap_key, AMAP_PLACEHOLDER)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"[完成] {path}: {len(data)} -> {len(kept)} 条（删除 {dropped} 条污染样本）")


def main():
    files = sys.argv[1:] or DEFAULT_FILES
    amap_key = load_amap_key()
    if not amap_key:
        print("[警告] .env 中未找到 AMAP_KEY，仅执行污染样本剔除")
    for rel in files:
        clean_file(os.path.join(DATA_DIR, rel), amap_key)
    print("全部完成。")


if __name__ == "__main__":
    main()
