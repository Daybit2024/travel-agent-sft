"""技能 token 估算(粗略,cl100k_base):default 完整 prompt vs 技能模式。

用法:
    python -m agent_graph.skill_token_check
"""
import os
import sys

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import tiktoken

from agent_graph.prompts import build_system_prompt
from agent_graph.skills import build_skill_system_prompt, load_skills

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ENC = tiktoken.get_encoding("cl100k_base")


def count(text: str) -> int:
    return len(ENC.encode(text))


def main():
    skills = load_skills()
    listing = "\n".join(f"- {s.name}: {s.description}" for s in skills.values())
    default_tokens = count(build_system_prompt({}))
    print(f"default 技能(原完整 prompt): {default_tokens} tokens")

    for name, skill in skills.items():
        skill_prompt = build_skill_system_prompt(name, {})
        print(f"技能 {name}: {count(skill_prompt)} tokens"
              f"(其中正文 {count(skill.body)} / 清单 {count(listing)})")
        print(f"  相对 default 降幅: {100 * (1 - count(skill_prompt) / default_tokens):.1f}%")


if __name__ == "__main__":
    main()
