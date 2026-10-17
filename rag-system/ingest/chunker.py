# -*- coding: utf-8 -*-
"""攻略文本切块器（small-to-big 的"小"）。

策略：
  - 去掉文件头部元数据行（城市编码/生成时间/=====），保留标题结构
  - 按 "## 小节" 切块；过小的小节并入相邻块；过长的小节按 "###" 子标题
    或空行段落继续下切，并保留重叠窗口
  - 每块记录父文档信息（city_code 等），供检索命中后放大回整篇
"""
import re

from config import CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, CHUNK_OVERLAP_CHARS

_META_RE = re.compile(r"^(城市编码|生成时间|生成用时|生成模型)\s*[:：]")


def _clean_lines(text: str):
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("="):
            continue
        if _META_RE.match(s):
            continue
        out.append(ln.rstrip())
    return out


def _split_sections(lines):
    """按 '## ' 切小节；首个 '##' 之前的内容归入默认小节"""
    sections = []
    cur_title, cur = "简介", []
    for ln in lines:
        if ln.startswith("## "):
            sections.append((cur_title, "\n".join(cur).strip()))
            cur_title, cur = ln[3:].strip(), []
        else:
            cur.append(ln)
    sections.append((cur_title, "\n".join(cur).strip()))
    return [(t, b) for t, b in sections if b]


def _split_long(text: str):
    """过长文本：优先按 '### ' 子标题切，否则按空行段落切；组装时带重叠"""
    if len(text) <= CHUNK_MAX_CHARS:
        return [text]
    parts = re.split(r"(?m)^(?=### )", text)
    if len(parts) <= 1:
        parts = re.split(r"\n\s*\n", text)
    blocks = [p.strip() for p in parts if p.strip()]
    chunks, cur = [], ""
    for b in blocks:
        if cur and len(cur) + len(b) + 1 > CHUNK_MAX_CHARS:
            chunks.append(cur)
            tail = cur[-CHUNK_OVERLAP_CHARS:] if CHUNK_OVERLAP_CHARS else ""
            cur = (tail + "\n" + b).strip()
        else:
            cur = (cur + "\n" + b).strip() if cur else b
    if cur:
        chunks.append(cur)
    return chunks


def chunk_guide(text: str, city_code: str, city_name: str, province_name: str, source_file: str):
    """把一篇攻略切成块列表；每块含父文档信息并可直接入库"""
    lines = _clean_lines(text)
    sections = _split_sections(lines)

    # 去掉纯标题残段（如文件头只剩标题行）
    sections = [(t, b) for t, b in sections if len(b) >= 60]

    # 合并过小的小节：优先并入下一节（保持阅读顺序），末节并入前一节
    merged = []
    i = 0
    while i < len(sections):
        title, body = sections[i]
        if len(body) < CHUNK_MIN_CHARS and i + 1 < len(sections):
            nt, nb = sections[i + 1]
            sections[i + 1] = (nt, f"## {title}\n{body}\n\n" + (f"## {nt}\n" if nt else "") + nb)
        elif len(body) < CHUNK_MIN_CHARS and merged:
            pt, pb = merged[-1]
            merged[-1] = (pt, pb + "\n\n" + (f"## {title}\n" if title else "") + body)
        else:
            merged.append((title, body))
        i += 1

    chunks = []
    idx = 0
    for title, body in merged:
        # 前向合并的块，body 开头已含小节标题行（## ...），不再重复前置
        composed = body if body.startswith("## ") else (f"## {title}\n{body}" if title else body)
        for piece in _split_long(composed):
            chunks.append({
                "chunk_id": f"{city_code}_{idx:03d}",
                "city_code": city_code,
                "city_name": city_name,
                "province_name": province_name,
                "source_file": source_file,
                "chunk_index": idx,
                "title": title,
                "content": piece,
            })
            idx += 1
    return chunks
