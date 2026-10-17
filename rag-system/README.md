# 旅游攻略 RAG 系统

面向旅行助手的攻略检索服务：**向量 + BM25 混合检索 → RRF 融合 → small-to-big 拼回整篇**，
并提供地名归一化（别称映射表 → 本地 4B 改写 → 语料清单校验 → 联网兜底）。

## 语料与规模

| 项 | 值 |
|---|---|
| 攻略篇数 | **731 篇**（34 省，含 4 直辖市与 387 县级市）|
| 切块后 | **5502 块**（平均 7.5 块/篇，按小节的标题切分）|
| 嵌入 | 阿里云百炼 `text-embedding-v4`，1024 维，COSINE |
| 存储 | Milvus Lite（单文件 `milvus_lite.db`），集合名 **`travel_guides_chunks`** |

攻略文本由 `../data/generate_travel_guides.py` 调用 deepseek-v3 生成，
命名格式 `{城市编码}_{城市名}_travel_guide.txt`（如 `110000_北京_travel_guide.txt`）。

## 检索流程

```
query
 ├─ ① 别称映射表（location_aliases.json）："长安"→"西安"，确定性、0ms
 ├─ ② 表未识别 → 本地 4B 改写（query_rewrite.py，keep_alive 常驻 + 磁盘缓存）
 ├─ ③ 候选集过滤（城市编码）→ 向量 + BM25 双路召回 → RRF 融合 → 阈值
 ├─ ④ small-to-big：命中块上浮到正文最前，其余按文档序拼回整篇
 └─ 语料外地点（景点级 / 未覆盖城市）→ 联网兜底（博查），结果标注未核实
```

`search_strategy` 会如实返回走了哪条路：`location_priority`（本地检索）、
`web_fallback`（联网兜底）、`llm_rewrite`、`*_fallback`（未识别地点时的全库检索）。

## 快速开始

```bash
# 1. 依赖（在项目根目录的 .venv 中）
pip install flask pymilvus openai python-dotenv jieba

# 2. 导入语料（需要 DASHSCOPE_API_KEY 环境变量）
python ingest/importer.py            # 全量重建：先 drop 再 create
python ingest/importer.py --limit 20 # 试跑前 20 篇

# 3. 启动服务（默认 127.0.0.1:8010；也可用 ../scripts/start_rag.bat）
python rag_api.py

# 4. 回归评测（语料覆盖 + 联网兜底）
python eval/test_city_coverage.py
```

> **导入是全量重建，不支持增量**：`build_collection()` 会先 `drop_collection` 再重建，
> `collect_chunks()` 遍历整个语料目录。所以**导入前必须先停服务**（Milvus Lite 是单进程文件库）。
> 导入耗时约 35 秒（5502 块 / 8 并发）。新增语料后需同时更新
> `city_code_mapping.json`（`ingest/importer.py` 从它取城市名与省份）。

## API

### `GET /health`

```json
{"status": "healthy", "service": "RAG Travel Guide API",
 "collection": "travel_guides_chunks", "collection_size": 5502}
```

### `POST /search`

```json
{"query": "广州有什么好玩的", "search_type": "hybrid", "limit": 3,
 "vector_weight": 1.0, "keyword_weight": 1.0}
```

响应含 `search_strategy`、`matched_locations`（识别出的城市编码）与 `results[]`
（每条含 `city_name` / `matched_chunk` / `content` / `rrf_score` / `bm25_rank` / `vector_rank`）。

> `keyword_weight` 网格搜索结论：**1.5 是唯一会让块级排序翻车的配置**，默认 1.0。

### `POST /search_by_location`、`GET /stats`

按省/市过滤、查看集合统计（见 `rag_api.py`）。

## 配置

密钥一律走环境变量，不写进代码：

| 变量 | 用途 | 默认 |
|---|---|---|
| `DASHSCOPE_API_KEY` | 嵌入模型 | 必填 |
| `RAG_MILVUS_URI` | Milvus 库路径 | `rag-system/milvus_lite.db` |
| `RAG_REWRITE_BASE_URL` / `RAG_REWRITE_MODEL` | 本地 4B 改写 | `localhost:11434/v1` / `qwen3-travel` |
| `WEB_SEARCH_PROVIDER` / `BOCHA_API_KEY` | 联网兜底 | 空（则用 mock）|
| `RAG_REWRITE_ENABLED` / `RAG_STANDARDIZE_ENABLED` | 关掉 4B 两层 | `1` |

## 请求流水日志（可观测性）

每次检索落一行 JSONL 到 `logs/rag/YYYY-MM-DD.jsonl`（`request_log.py`）：query（脱敏截断）、
策略、命中城市、返回块的 `chunk_id` / 标题 / RRF 名次 / 向量与 BM25 名次，以及**正文长度**
——**只记结构字段，不记正文**（块级用 ID 定位，需要原文时离线重跑切块器即可，切块是确定性的）。

`trace_id` 由 agent 通过请求头 `X-Trace-Id` 传入，因此同一次用户请求在 agent 侧与 RAG 侧的事件
可以**跨进程串起来**；RAG 被独立调用时也会自己生成一个。

```bash
python ../scripts/request_log_report.py --dir logs/rag
# 输出：检索成功率、延迟 P50/P95、策略分布、联网兜底比例、块级命中明细
```

关闭：`REQUEST_LOG=0`。

## 已知限制

- **评测口径**：`eval/test_city_coverage.py` 的 A 组 Top1/Top3 是**结构性保证**，
  不是检索准确率——同一份城市映射表既生成候选集、又定义语料清单，且结果按城市聚合
  （该城任意一块过阈值即命中）。真正缺的是**块级/多城市排序**用例（现有块级用例仅 4 条）。
- **联网兜底覆盖面**：只对"未覆盖城市"与名单内景点生效；名单外的地点（如境外城市）
  会退化成全库检索，可能返回无关内容。本项目是国内旅行助手，出境游属设计外场景。
- **语料是静态快照**：门票、临时闭馆、季节性活动查不到，需靠联网兜底。
- 3 个已撤销的行政区划老名字查不到 adcode（惠阳市/海丰市/米泉市），未收录。

## 许可证

本项目仅供学习和研究使用。
