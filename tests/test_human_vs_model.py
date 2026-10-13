"""交接前自检：人类(模拟点击) vs 真实模型，走一遍浏览器会走的链路。

    python -m gomoku_arena.server --port 8090
    python tests/test_human_vs_model.py

与 test_human.py 的区别：对手是真实模型（DSH / Codex，取决于配置），
用来确认"人机对弈"这条链路在真实权重下也能跑起来。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"
CONFIG = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "matchups" / "play_human_vs_dsh.json")
MOVES_TO_PLAY = 4

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    ok = bool(ok)
    results.append((ok, name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post(path: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def main() -> int:
    print(f"=== 人机对弈链路自检（{CONFIG.split(chr(92))[-1]}）===\n")
    try:
        get("/api/engines")
    except Exception as exc:
        print(f"服务端不可达: {exc}")
        return 1

    status, snap = post("/api/matches", {
        "participants": {
            "black": {"engine": "human", "pid": "you", "name": "我"},
            "white": {"engine": __import__("json").loads(
                Path(CONFIG).read_text(encoding="utf-8"))["white"]["engine"],
                "pid": "model",
                "checkpoint": __import__("json").loads(
                    Path(CONFIG).read_text(encoding="utf-8"))["white"]["checkpoint"],
                "version": "v1", "name": "对手"},
        },
        "move_timeout_ms": 120000, "game_timeout_ms": 600000, "seed": 31337,
    })
    game_id = snap["game_id"]
    check(status == 202, "人机对局创建成功", game_id)

    played = 0
    deadline = time.time() + 300
    while played < MOVES_TO_PLAY and time.time() < deadline:
        state = get(f"/api/state?game_id={game_id}")
        if not state.get("running"):
            break
        if state.get("turn") != "black":       # 人类执黑
            time.sleep(0.2)
            continue
        occupied = {(m["x"], m["y"]) for m in state.get("moves", [])}
        pick = next(((x, y) for x in range(4, 11) for y in range(4, 11)
                     if (x, y) not in occupied), None)
        if pick is None:
            break
        code, _ = post(f"/api/human/black/{game_id}/move", {"x": pick[0], "y": pick[1]})
        if code != 202:
            time.sleep(0.3)
            continue
        before = len(occupied)
        for _ in range(200):                    # 等这手 + 模型回手
            cur = get(f"/api/state?game_id={game_id}")
            if len(cur.get("moves", [])) >= before + 2 or not cur.get("running"):
                break
            time.sleep(0.1)
        played += 1

    final = get(f"/api/state?game_id={game_id}")
    moves = final.get("moves", [])
    model_moves = [m for m in moves if m["color"] != "black"]
    check(played >= 2, "人类落子能被适配器转交并进入棋谱",
          f"模拟点击 {played} 手，棋谱 {len(moves)} 手")
    check(len(model_moves) >= 1, "真实模型做出了应答（人机对局在推进）",
          f"模型落子 {len(model_moves)} 手，耗时 {[m['ms'] for m in model_moves][:5]}")
    check(all(0 <= m["x"] < 15 and 0 <= m["y"] < 15 for m in moves),
          "棋谱坐标全部合法")
    print(f"       [状态] running={final.get('running')} "
          f"reason={(final.get('result') or {}).get('reason')} moves={len(moves)}")

    failed = [r for r in results if not r[0]]
    print(f"\n合计 {len(results)} 项，通过 {len(results) - len(failed)}，失败 {len(failed)}")
    for _, name, detail in failed:
        print(f"  FAIL: {name} — {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
