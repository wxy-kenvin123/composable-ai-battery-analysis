"""Trim bibliography/reference sections from extracted full-text article files.

Standalone helper recording the boundary rules used on the extracted article texts
(which are not redistributed). Reads ``*.txt`` from ``--input-dir`` and writes
trimmed texts to ``--output-dir``.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple


OVERRIDES = {
    "10414": "REFER ENCES 1.", "10440": "2. Xu K. Electrolytes",
    "10478": "[1] S. Hameer", "15071": "REFERENCES", "20296": "REFER ENCES 1.",
    "20333": "[I] B. Serosati", "20370": "[D] J. Zhang", "20374": "I REFERENCES ()",
    "20388": "[i] A. Urban", "20394": "(1) 8. Esser", "20396": "Tl] ) 5. Froyen",
    "30378": "[1] Hf. Deiseroth", "30383": "{) LQ Mu", "35046": "[1] B. Li, K. Zhang",
    "35056": "[17] Y. Yuan",
}


def truncate_after_references(raw_text: str, min_ratio: float = 0.35, min_chars: int = 4000) -> Tuple[str, Optional[str]]:
    raw = raw_text.replace("\f", "\n")
    threshold = max(min_chars, int(min_ratio * len(raw)))
    cand = []
    pat = r"(?im)^[\s#>*-]*?(?:\d+[.\s]+)?(?:notes\s+and\s+references|references\s+and\s+notes|reference[s]?|bibliography|literature cited)\s*[:.]?\s*$"
    for m in re.finditer(pat, raw):
        if m.start() >= threshold:
            cand.append((m.start(), "reference_heading"))
    for m in re.finditer(r"(?m)^\s*(?:\[\s*1\s*\]|\(1\)|1[.)])\s+", raw):
        if m.start() < threshold:
            continue
        tail = raw[m.start():m.start() + 7000]
        nums = set(re.findall(r"(?m)^\s*(?:\[\s*(\d+)\s*\]|\((\d+)\)|(\d+)[.)])\s+", tail))
        years = re.findall(r"\b(?:19|20)\d{2}\b", tail)
        if len(nums) >= 6 and len(years) >= 6:
            cand.append((m.start(), "numbered_reference_block_with_years"))
    for m in re.finditer(r"(?i)\b(?:references|bibliography|notes and references)\s*[:.]?\s*(?=\(1\)|\[1\]|1[.)]\s)", raw):
        if m.start() < threshold:
            continue
        tail = raw[m.start():m.start() + 7000]
        nums = set(re.findall(r"(?:\[(\d+)\]|\((\d+)\))", tail))
        if len(nums) >= 6 and len(re.findall(r"\b(?:19|20)\d{2}\b", tail)) >= 6:
            cand.append((m.start(), "inline_reference_heading_with_citations"))
    if cand:
        pos, rule = min(cand)
        return raw[:pos].rstrip(), rule
    return raw, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(a.input_dir.glob("*.txt")):
        trimmed, _ = truncate_after_references(src.read_text(encoding="utf-8", errors="replace"))
        (a.output_dir / src.name).write_text(trimmed, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
