# -*- coding: utf-8 -*-
"""旧导入脚本（整篇嵌入）已由 ingest/importer.py 取代（小节级切块 + chunk 集合）。

保留本文件仅为兼容旧用法，执行会转发到新导入管线：
    python -m ingest.importer            # 全量重建
    python -m ingest.importer --limit 5  # 小样本试跑
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("[提示] 旧整篇导入已弃用，转用 ingest.importer（小节级切块）")
    runpy.run_module("ingest.importer", run_name="__main__")
