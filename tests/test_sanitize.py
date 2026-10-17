# -*- coding: utf-8 -*-
"""数据清洗规则（training/sanitize_dataset.py）"""
import io
import json

from sanitize_dataset import CONTAMINATION_MARK, clean_file

AMAP = "ABCDEF1234567890"


def _write(tmp_path, samples):
    p = tmp_path / "data.json"
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    return str(p)


def test_drops_contaminated_sample_and_masks_key(tmp_path):
    samples = [
        {"conversation": [{"role": "user", "content": "正常样本"}]},
        {"conversation": [{"role": "assistant", "content": json.dumps({"arguments": {CONTAMINATION_MARK: "xxx"}})}]},
        {"conversation": [{"role": "tool", "content": f"出错: ...&key={AMAP}"}]},
    ]
    path = _write(tmp_path, samples)
    clean_file(path, AMAP)

    data = json.load(io.open(path, encoding="utf-8"))
    assert len(data) == 2
    text = json.dumps(data, ensure_ascii=False)
    assert CONTAMINATION_MARK not in text
    assert AMAP not in text
    assert "REDACTED_AMAP_KEY" in text


def test_clean_is_idempotent(tmp_path):
    samples = [{"conversation": [{"role": "tool", "content": f"key={AMAP}"}]}]
    path = _write(tmp_path, samples)
    clean_file(path, AMAP)
    clean_file(path, AMAP)

    data = json.load(io.open(path, encoding="utf-8"))
    assert len(data) == 1
