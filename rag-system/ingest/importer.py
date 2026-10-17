# -*- coding: utf-8 -*-
"""入库管线：攻略文件 → 切块 → 嵌入 → Milvus（chunk 级集合）。

用法（在 rag-system 目录下执行）：
    python -m ingest.importer            # 全量重建（先 drop 旧集合）
    python -m ingest.importer --limit 5  # 小样本试跑

注意：milvus-lite 同一时间只能被一个进程打开——导入前先停掉 rag_api 服务。
"""
import argparse
import glob
import io
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from pymilvus import DataType, MilvusClient
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (BASE_DIR, COLLECTION_NAME, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
                    EMBEDDING_DIM, EMBEDDING_MODEL, GUIDES_DIR, METRIC_TYPE, MILVUS_URI)
from ingest.chunker import chunk_guide

BATCH_SIZE = 10     # 每次嵌入请求的文本条数
MAX_WORKERS = 8     # 嵌入并发线程数


def load_city_meta():
    """加载 adcode → {city_name, province_name}"""
    with io.open(os.path.join(BASE_DIR, "city_code_mapping.json"), encoding="utf-8") as f:
        data = json.load(f)
    meta = {}
    for adcode, info in (data.get("城市编码映射") or {}).items():
        if isinstance(info, dict) and info.get("name"):
            meta[str(adcode)] = {"city_name": info["name"], "province_name": info.get("province", "")}
    return meta


def collect_chunks(limit=0):
    files = sorted(glob.glob(os.path.join(GUIDES_DIR, "*_travel_guide.txt")))
    if limit:
        files = files[:limit]
    meta = load_city_meta()
    chunks = []
    for path in files:
        base = os.path.basename(path)
        code, city = base.split("_", 1)[0], base.split("_")[1]
        m = meta.get(code, {})
        text = io.open(path, encoding="utf-8").read()
        chunks.extend(chunk_guide(text, code, m.get("city_name", city),
                                  m.get("province_name", ""), base))
    return chunks


def embed_texts(client, texts):
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIM,
        encoding_format="float",
    )
    return [d.embedding for d in resp.data]


def build_collection(client):
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("city_code", DataType.VARCHAR, max_length=16)
    schema.add_field("city_name", DataType.VARCHAR, max_length=64)
    schema.add_field("province_name", DataType.VARCHAR, max_length=64)
    schema.add_field("source_file", DataType.VARCHAR, max_length=256)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("title", DataType.VARCHAR, max_length=256)
    schema.add_field("content", DataType.VARCHAR, max_length=8192)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="FLAT", metric_type=METRIC_TYPE)
    client.create_collection(COLLECTION_NAME, schema=schema, index_params=index_params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只导入前 N 篇（试跑用）")
    args = ap.parse_args()

    if not DASHSCOPE_API_KEY:
        print("[错误] 缺少 DASHSCOPE_API_KEY")
        return

    chunks = collect_chunks(args.limit)
    print(f"共 {len(chunks)} 个块待导入")

    client = MilvusClient(uri=MILVUS_URI)
    build_collection(client)
    print(f"集合已重建: {COLLECTION_NAME}（dim={EMBEDDING_DIM}, metric={METRIC_TYPE}）")

    lock = threading.Lock()
    done = {"n": 0}
    openai_client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

    def embed_batch(batch):
        vecs = embed_texts(openai_client, [c["content"] for c in batch])
        rows = []
        for c, v in zip(batch, vecs):
            row = dict(c)
            row["embedding"] = v
            rows.append(row)
        with lock:
            client.insert(COLLECTION_NAME, data=rows)
            done["n"] += len(rows)

    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(embed_batch, b) for b in batches]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="嵌入并入库"):
            pass

    stats = client.get_collection_stats(COLLECTION_NAME)
    print(f"完成：集合 {COLLECTION_NAME} 共 {stats.get('row_count', '?')} 个块")


if __name__ == "__main__":
    main()
