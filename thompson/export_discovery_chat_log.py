#!/usr/bin/env python3
"""
Export the complete interactive research & discovery chat log for the D'Arcy
Thompson 1917 Fish Transformations & Morphospace study into both readable
Markdown (docs/THOMPSON_1917_DISCOVERY_CHAT_LOG.md) and structured JSONL
(docs/thompson_1917_discovery_chat_log.jsonl).
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT_PATH = Path(
    "/Users/jdietz/.cursor/projects/Users-jdietz-Documents-GitHub-infinitemorphospace/"
    "agent-transcripts/0e50eb21-062e-47c3-9ded-1b0bbe630bee/"
    "0e50eb21-062e-47c3-9ded-1b0bbe630bee.jsonl"
)
OUT_MD = REPO_ROOT / "docs" / "THOMPSON_1917_DISCOVERY_CHAT_LOG.md"
OUT_JSONL = REPO_ROOT / "docs" / "thompson_1917_discovery_chat_log.jsonl"


def main():
    lines = [
        json.loads(line)
        for line in TRANSCRIPT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    md_out = [
        "# Full Research & Discovery Chat Log: Resolving D'Arcy Thompson's 1917 Fish Transformations Paradox",
        "",
        "This document records the complete interactive research, mathematical modeling, multi-agent referee auditing, "
        "and domain-expert feedback dialogue (including feedback from Prof. Peter C. Wainwright, UC Davis) that led to "
        "the discoveries documented in [`docs/KEY_DISCOVERIES.md`](./KEY_DISCOVERIES.md) and "
        "[`docs/thompson_1917_fish_transformations_manuscript.pdf`](./thompson_1917_fish_transformations_manuscript.pdf).",
        "",
        "---",
        "",
    ]

    clean_jsonl = []
    turn_num = 0
    last_user_q = None

    for obj in lines:
        role = obj.get("role", "")
        msg = obj.get("message", {})
        content = msg.get("content", [])
        if role == "user":
            raw_txt = "\n".join(
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
            m = re.search(r"<user_query>(.*?)</user_query>", raw_txt, re.S)
            if not m:
                continue
            q = m.group(1).strip()
            if not q or q == last_user_q or q.startswith("Briefly inform the user"):
                continue
            last_user_q = q
            turn_num += 1
            ts_m = re.search(r"<timestamp>(.*?)</timestamp>", raw_txt)
            ts_str = f" ({ts_m.group(1).strip()})" if ts_m else ""
            md_out.append(f"## Turn {turn_num} — User{ts_str}")
            md_out.append("")
            md_out.append(q)
            md_out.append("")
            clean_jsonl.append(
                {
                    "turn": turn_num,
                    "role": "user",
                    "timestamp": ts_m.group(1).strip() if ts_m else None,
                    "text": q,
                }
            )
        elif role == "assistant":
            texts = []
            tools = []
            for c in content:
                if not isinstance(c, dict):
                    continue
                ctype = c.get("type")
                if ctype == "text" and c.get("text", "").strip():
                    t = c["text"].strip()
                    t = t.replace(
                        "/Users/jdietz/Documents/GitHub/infinitemorphospace/docs/",
                        "./",
                    )
                    t = t.replace(
                        "/Users/jdietz/Documents/GitHub/infinitemorphospace/",
                        "../",
                    )
                    texts.append(t)
                elif ctype == "tool_use":
                    tname = c.get("name", "tool")
                    tinp = c.get("input", {})
                    desc = (
                        tinp.get("description")
                        or tinp.get("path")
                        or str(tinp.get("command", ""))[:80]
                    )
                    tools.append(f"`{tname}` ({desc})")
            if tools and not texts:
                clean_jsonl.append(
                    {"turn": turn_num, "role": "assistant_tool", "tools": tools}
                )
            elif texts:
                full_a = "\n\n".join(texts)
                md_out.append(f"### Assistant Response (Turn {turn_num})")
                md_out.append("")
                if tools:
                    md_out.append("*Tools executed:* " + ", ".join(tools))
                    md_out.append("")
                md_out.append(full_a)
                md_out.append("")
                md_out.append("---")
                md_out.append("")
                clean_jsonl.append(
                    {
                        "turn": turn_num,
                        "role": "assistant",
                        "tools": tools,
                        "text": full_a,
                    }
                )

    OUT_MD.write_text("\n".join(md_out), encoding="utf-8")
    OUT_JSONL.write_text(
        "\n".join(json.dumps(r) for r in clean_jsonl) + "\n", encoding="utf-8"
    )
    print(f"Exported {turn_num} turns ({len(clean_jsonl)} records) to {OUT_MD} and {OUT_JSONL}")


if __name__ == "__main__":
    main()
