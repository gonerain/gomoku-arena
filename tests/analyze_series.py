"""分析一批棋谱：唯一性、颜色与胜负的关系、手数分布。

    python tests/analyze_series.py records/games.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "records/games.jsonl")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        print("没有棋谱")
        return 1

    print(f"棋谱: {path}")
    print(f"总局数: {len(records)}")
    print()

    # 颜色 vs 胜负
    by_color = Counter()
    for rec in records:
        by_color[(rec["result"]["winner"], rec["black"]["pid"])] += 1
    print("胜负与执黑方的关系:")
    for (winner, black_pid), count in sorted(by_color.items(), key=lambda kv: str(kv[0])):
        label = winner if winner else "无胜者"
        print(f"  黑方={black_pid:<6} → 胜者={label:<6} : {count} 局")
    print()

    # 手数分布
    lengths = Counter(len(rec["moves"]) for rec in records)
    print("手数分布:", dict(sorted(lengths.items())))
    print()

    # 着法序列唯一性（分别看 DSH 执黑、Codex 执黑的局面）
    unique = defaultdict(set)
    for rec in records:
        key = rec["black"]["pid"]
        seq = tuple((m["x"], m["y"]) for m in rec["moves"])
        unique[key].add(seq)
    print("着法序列唯一性:")
    for pid, seqs in unique.items():
        total = sum(1 for rec in records if rec["black"]["pid"] == pid)
        print(f"  {pid} 执黑: {total} 局，其中不同序列 {len(seqs)} 条")
    print()

    # 逐手耗时统计（判断引擎有没有在思考）
    print("单步耗时（ms）:")
    for pid in ("dsh", "codex"):
        samples = [
            m["ms"] for rec in records for m in rec["moves"]
            if rec["black" if m["color"] == "black" else "white"]["pid"] == pid
        ]
        if samples:
            samples_sorted = sorted(samples)
            n = len(samples_sorted)
            print(f"  {pid:<6} n={n:<5} 中位={samples_sorted[n // 2]:<6} "
                  f"最大={samples_sorted[-1]:<6} 均值={sum(samples) / n:.1f}")
    print()

    # seed 是否真的影响棋局
    seeds = [(rec.get("seed"), len(rec["moves"])) for rec in records]
    print("seed 与手数:", seeds[:6], "..." if len(seeds) > 6 else "")
    same_color_seeds = defaultdict(list)
    for rec in records:
        same_color_seeds[rec["black"]["pid"]].append(rec.get("seed"))
    for pid, values in same_color_seeds.items():
        print(f"  {pid} 执黑的 seed 序列: {values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
