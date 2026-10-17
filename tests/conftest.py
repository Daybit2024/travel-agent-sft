# -*- coding: utf-8 -*-
"""测试公共配置：无密钥环境下也能导入项目模块。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 单元测试不触网：设置占位 key，避免 get_route 导入时校验报错
os.environ.setdefault("AMAP_KEY", "test-key-for-import")

# 单元测试不写请求流水（否则每次 pytest 都会往 logs/ 里灌数据）
os.environ.setdefault("REQUEST_LOG", "0")

for _p in (ROOT, ROOT / "agent", ROOT / "training", ROOT / "eval"):
    sys.path.insert(0, str(_p))
