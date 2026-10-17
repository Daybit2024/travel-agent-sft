# -*- coding: utf-8 -*-
"""RAG 组件集中配置：路径 / 切块 / 检索参数。

调参只改这里；环境变量优先（部署时可用环境变量覆盖）。
密钥与开关放项目根 .env（不入库），与 Agent 侧共用一份。
"""
import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根 .env（系统环境变量优先，.env 作补充）
load_dotenv(os.path.join(os.path.dirname(BASE_DIR), ".env"))

# ── 路径与模型 ──────────────────────────────────────────────
MILVUS_URI = os.environ.get("RAG_MILVUS_URI", os.path.join(BASE_DIR, "milvus_lite.db"))
COLLECTION_NAME = "travel_guides_chunks"
GUIDES_DIR = os.path.join(BASE_DIR, "original_data", "travel_guides")

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024
METRIC_TYPE = "COSINE"

# ── 切块（small-to-big 的"小"）──────────────────────────────
CHUNK_MIN_CHARS = 100       # 小于此长度的小节并入相邻块（攻略小节多为 100~300 字符，是天然检索单元）
CHUNK_MAX_CHARS = 1200      # 大于此长度的小节继续下切（按子标题/段落）
CHUNK_OVERLAP_CHARS = 100   # 下切时的重叠窗口

# ── 检索 ───────────────────────────────────────────────────
RECALL_LIMIT = 20           # 每路（向量 / BM25）召回的候选块数
VECTOR_SIM_THRESHOLD = 0.30 # 向量相似度下限（COSINE）
BM25_SCORE_THRESHOLD = 0.5  # BM25 分数下限（词频尺度，视语料校准）
RRF_K = 60                  # RRF 倒数排名融合参数（经典值）
RRF_VECTOR_WEIGHT = 1.0
RRF_KEYWORD_WEIGHT = 1.0    # 网格搜索结论：1.5 会把 BM25 的噪声放大导致块级错排（eval/grid_search_weights.py）
PARENT_MAX_CHARS = 4000     # small-to-big 返回父篇（整篇攻略）时的字符上限

# ── 网络检索兜底（本地语料未覆盖的地点）──────────────────────
WEB_SEARCH_PROVIDER = os.environ.get("WEB_SEARCH_PROVIDER", "")  # bocha / tavily；空则用 mock
WEB_SEARCH_LIMIT = 3
WEB_SEARCH_TIMEOUT = 10
# 联网查询统一补的意图词：裸地名（如"西安"）直搜会命中百科/新闻，
# 补上旅游意图后结果才能稳定落在攻略内容上
WEB_QUERY_SUFFIX = os.environ.get("WEB_QUERY_SUFFIX", " 旅游攻略")

# ── Query 改写与标准化（本地 4B）────────────────────────────
REWRITE_ENABLED = os.environ.get("RAG_REWRITE_ENABLED", "1") != "0"
REWRITE_BASE_URL = os.environ.get("RAG_REWRITE_BASE_URL", "http://localhost:11434/v1")
REWRITE_MODEL = os.environ.get("RAG_REWRITE_MODEL", "qwen3-travel")
REWRITE_TIMEOUT = 20
# 语言标准化：地名替换后、检索前，把 query 整理成规范检索式（每条查询 +~3s）
STANDARDIZE_ENABLED = os.environ.get("RAG_STANDARDIZE_ENABLED", "1") != "0"
