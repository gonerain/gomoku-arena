"""端到端接口自检：需要先启动服务端。

    python -m platform.server --port 8090
    python tests/test_api.py

覆盖浏览器实际会走的全部路径：列表、开新局、轮询实时状态、结束后回放。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"
results: list[tuple[bool, str, str]] = []


def record(ok: bool, name: str, detail: str = "") -> None:
    results.append((ok, name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post(path: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def main() -> int:
    # 1) 首页与静态资源
    try:
        with urllib.request.urlopen(BASE + "/", timeout=15) as resp:
            html = resp.read().decode("utf-8")
        record(resp.status == 200 and 'id="board"' in html and "/api/state" in html,
               "首页可访问且包含棋盘与轮询逻辑", f"{len(html)} 字节")
    except Exception as exc:
        record(False, "首页可访问", str(exc))
        return 1

    # 2) 引擎列表
    engines = get("/api/engines")
    keys = [e["key"] for e in engines]
    record(len(engines) >= 3 and "trained" in keys, "引擎列表含 trained 接口模板", ",".join(keys))

    # 3) 开新局
    status, snap = post("/api/matches", {
        "black": "greedy", "white": "random",
        "move_timeout_ms": 800, "game_timeout_ms": 60000, "seed": 4242,
    })
    game_id = snap.get("game_id", "")
    record(status == 202 and snap.get("running") is True and game_id.startswith("g-"),
           "开新局返回进行中快照", f"status={status} id={game_id}")

    # 4) 轮询实时状态直到结束
    trace, final = [], None
    for _ in range(150):
        state = get(f"/api/state?game_id={game_id}")
        trace.append((state.get("move_count"), state.get("status")))
        if not state.get("running"):
            final = state
            break
        time.sleep(0.1)
    running_seen = any(s == "running" for _, s in trace)
    finished_seen = final is not None and final.get("status") == "finished"
    record(running_seen, "轮询能观察到 running 中间态", f"{len(trace)} 次采样")
    record(finished_seen, "轮询能观察到 finished 终态",
           f"winner={final and final['result']['winner']} reason={final and final['result']['reason']}")

    # 5) 终态必须可从棋谱回放还原
    rec = get(f"/api/games/{game_id}")
    moves = rec.get("moves", [])
    legal = all(0 <= m["x"] < 15 and 0 <= m["y"] < 15 for m in moves)
    unique = len({(m["x"], m["y"]) for m in moves}) == len(moves)
    record(legal and unique and len(moves) > 0,
           "棋谱可回放：坐标合法且无重复落子", f"{len(moves)} 手")

    # 6) 元信息与训练预算字段
    meta = rec["black"].get("meta", {})
    record(bool(meta.get("algo")) and "version" in rec["black"],
           "记录含模型版本与训练预算(meta)", f"black.meta={json.dumps(meta, ensure_ascii=False)}")

    # 7) 列表含全部必要字段
    games = get("/api/games?limit=5")
    first = games[0]
    need = ("game_id", "ts", "black", "white", "result", "moves", "move_count", "status")
    missing = [k for k in need if k not in first]
    record(not missing, "对局列表字段齐全", "缺: " + ",".join(missing) if missing else "OK")

    # 8) 错误处理
    try:
        post("/api/matches", {"black": "no-such-engine", "white": "random"})
        record(False, "未知引擎被拒绝", "竟然成功了")
    except urllib.error.HTTPError as exc:
        record(exc.code == 400, "未知引擎返回 400", f"HTTP {exc.code}")
    try:
        get("/api/games/g-does-not-exist")
        record(False, "不存在的对局返回 404", "竟然成功了")
    except urllib.error.HTTPError as exc:
        record(exc.code == 404, "不存在的对局返回 404", f"HTTP {exc.code}")

    failed = [r for r in results if not r[0]]
    print(f"\n合计 {len(results)} 项，通过 {len(results) - len(failed)}，失败 {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
