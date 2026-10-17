#!/usr/bin/env python3
# coding: utf-8
"""运行 llama.cpp 的 convert_hf_to_gguf.py，但绕过 numpy tofile 在非零偏移时
误报 "Not enough free space" 的检查：改为直接 write(tobytes())，字节内容一致。
命令行参数与 convert_hf_to_gguf.py 完全相同。
"""

import runpy
import sys

CONVERTER = "/root/autodl-tmp/llama.cpp/convert_hf_to_gguf.py"
sys.path.insert(0, "/root/autodl-tmp/llama.cpp/gguf-py")
sys.path.insert(0, "/root/autodl-tmp/llama.cpp")

from gguf.lazy import LazyNumpyTensor


def _tofile_write(self, f, *args, **kwargs):
    eager = LazyNumpyTensor.to_eager(self)
    if hasattr(f, "write"):
        f.write(eager.tobytes())
        return None
    return eager.tofile(f, *args, **kwargs)


LazyNumpyTensor.tofile = _tofile_write
sys.argv[0] = CONVERTER
runpy.run_path(CONVERTER, run_name="__main__")
