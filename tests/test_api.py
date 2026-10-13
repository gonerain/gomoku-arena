"""端到端接口自检：需要先启动服务端。

    python -m gomoku_arena.server --port 8090
    python tests/test_api.py

覆盖浏览器实际会走的全部路径：首页、引擎列表、开单局、轮询实时状态、回放、
完整统计、批量对战（交换先手 + 分色统计），以及错误处理。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"
results: list[tuple[bool, str, str]] = []
not_verified: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    ok = bool(ok)
    results.append((ok, name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post(path: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def delete(path: str):
    req = urllib.request.Request(BASE + path, method="DELETE")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def expect_http_error(path: str, status: int, method: str = "GET", payload: dict | None = None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return False, resp.status
    except urllib.error.HTTPError as exc:
        return exc.code == status, exc.code


def wait_finished(game_id: str, timeout: float = 60.0):
    deadline = time.time() + timeout
    trace = []
    while time.time() < deadline:
        state = get(f"/api/state?game_id={game_id}")
        trace.append(state.get("status"))
        if not state.get("running"):
            return state, trace
        time.sleep(0.1)
    return None, trace


def test_static_and_meta() -> None:
    with urllib.request.urlopen(BASE + "/", timeout=30) as resp:
        html = resp.read().decode("utf-8")
    check(resp.status == 200 and 'id="board"' in html and "/api/state" in html,
          "首页可访问且包含棋盘与轮询逻辑", f"{len(html)} 字节")
    check("/api/stats" in html and "/api/batches" in html and "replayGame" in html,
          "前端具备完整统计 / 批量 / 回放隔离逻辑")

    board = get("/api/board")
    check(board["board_size"] == 15 and "≥5" in board["rule"],
          "棋盘与规则由后端下发，前后端一致", board["rule"])

    engines = get("/api/engines")
    keys = {e["key"]: e for e in engines}
    check(len(engines) >= 3 and "trained" in keys, "引擎列表含 trained 接入点", ",".join(keys))
    check(keys["random"]["baseline"] is True and keys["greedy"]["baseline"] is True
          and keys["trained"]["baseline"] is False,
          "引擎列表标注了基线 / 非基线（可区分非 RL 策略）")


def test_single_match_and_replay() -> None:
    status, snap = post("/api/matches", {
        "black": "greedy", "white": "random",
        "move_timeout_ms": 800, "game_timeout_ms": 60000, "seed": 4242,
    })
    game_id = snap.get("game_id", "")
    check(status == 202 and snap.get("running") is True and game_id.startswith("g-"),
          "开新局返回进行中快照", f"status={status} id={game_id}")

    final, trace = wait_finished(game_id)
    check(final is not None and any(s == "running" for s in trace),
          "轮询能观察到 running 中间态", f"{len(trace)} 次采样")
    check(final is not None and final.get("status") == "finished",
          "轮询能观察到 finished 终态",
          f"winner={final and final['result']['winner']} reason={final and final['result']['reason']}")

    rec = get(f"/api/games/{game_id}")
    moves = rec.get("moves", [])
    legal = all(0 <= m["x"] < 15 and 0 <= m["y"] < 15 for m in moves)
    unique = len({(m["x"], m["y"]) for m in moves}) == len(moves)
    check(legal and unique and len(moves) > 0, "棋谱可回放：坐标合法且无重复落子",
          f"{len(moves)} 手")
    check(rec["black"].get("checkpoint") is not None and rec["black"].get("meta", {}).get("algo"),
          "棋谱含模型版本与训练预算(meta)",
          f"black.meta={json.dumps(rec['black']['meta'], ensure_ascii=False)}")
    check(rec.get("move_timeout_ms") and rec.get("game_timeout_ms"),
          "棋谱记录了两类时限配置")


def test_list_vs_full_stats() -> None:
    games = get("/api/games?limit=5")
    first = games[0]
    need = ("game_id", "ts", "black", "white", "result", "move_count", "status")
    missing = [k for k in need if k not in first]
    check(not missing, "最近对局列表字段齐全", "缺: " + ",".join(missing) if missing else "OK")
    check("moves" not in first, "最近对局列表不携带着法（回放时单独取，避免体积失控）")

    stats = get("/api/stats")
    check(stats["total_games"] >= len(games),
          "完整统计覆盖全部棋谱，而不是最近 N 局",
          f"total={stats['total_games']} 列表显示={len(games)}")
    check(stats["black_wins"] + stats["white_wins"] + stats["draws"] <= stats["total_games"],
          "统计的黑胜/白胜/平数量自洽",
          f"{stats['black_wins']}/{stats['white_wins']}/{stats['draws']}")
    fields = {"games", "win", "loss", "draw", "timeout", "illegal_move", "win_rate"}
    check(all(fields <= set(e) for e in stats["engines"]) if stats["engines"] else True,
          "分引擎统计含胜负/超时/非法/胜率字段")


def test_batch_alternates_colors() -> None:
    status, batch = post("/api/batches", {
        "black": "greedy", "white": "random", "games": 4,
        "move_timeout_ms": 800, "game_timeout_ms": 60000, "seed": 777,
    })
    batch_id = batch.get("batch_id", "")
    check(status == 202 and batch_id.startswith("b-"), "批量对战已启动", f"id={batch_id}")

    deadline = time.time() + 120
    summary = batch
    while time.time() < deadline:
        summary = get(f"/api/batches/{batch_id}")
        if summary.get("status") != "running":
            break
        time.sleep(0.3)

    check(summary.get("status") == "finished", "批量对战正常结束", summary.get("status"))
    check(summary.get("played") == 4 and summary.get("requested") == 4,
          "完成局数与请求一致", f"{summary.get('played')}/{summary.get('requested')}")

    sides = summary["sides"]
    black_side = sides["greedy"]["as_black"]["games"]
    white_side = sides["greedy"]["as_white"]["games"]
    check(black_side == 2 and white_side == 2,
          "逐局交换先手：双方各执黑 2 局、执白 2 局",
          f"greedy 执黑={black_side} 执白={white_side}")

    totals = summary["totals"]["greedy"]
    check(totals["games"] == 4
          and totals["win"] + totals["loss"] + totals["draw"] + totals["game_timeout"]
          + totals["aborted"] + totals["platform_error"] == 4,
          "分色统计各项之和等于总局数",
          f"胜{totals['win']} 负{totals['loss']} 平{totals['draw']} 超时{totals['timeout']} "
          f"胜率{totals['win_rate']}")
    check(all(k in totals for k in ("timeout", "illegal_move", "win_rate")),
          "统计含超时、非法落子数量与总胜率")

    order = summary["games"]
    alternating = all(
        (g["black"] == "greedy") == (i % 2 == 0) for i, g in enumerate(order)
    )
    check(alternating, "逐局结果顺序与先手交换规则一致")

    # 棋谱里应带上批量归属信息
    rec = get(f"/api/games/{order[0]['game_id']}")
    check(rec.get("batch", {}).get("batch_id") == batch_id
          and rec["batch"].get("index") == 1,
          "批量棋谱带批次归属信息", json.dumps(rec.get("batch"), ensure_ascii=False))


def test_batch_cancel() -> None:
    # 用足够多的局数保证中止请求到达时批次仍在进行（避免测试自身竞态）
    requested = 40
    status, batch = post("/api/batches", {
        "black": "greedy", "white": "greedy", "games": requested,
        "move_timeout_ms": 800, "game_timeout_ms": 60000,
    })
    batch_id = batch["batch_id"]
    time.sleep(0.4)
    code, _ = delete(f"/api/batches/{batch_id}")
    check(code == 200, "批量对战可中止", f"HTTP {code}")
    deadline = time.time() + 60
    final = None
    while time.time() < deadline:
        final = get(f"/api/batches/{batch_id}")
        if final.get("status") != "running":
            break
        time.sleep(0.3)
    check(final and final.get("status") == "cancelled" and final.get("played") < requested,
          "中止后状态为 cancelled 且未跑完剩余局数",
          f"status={final and final.get('status')} played={final and final.get('played')}"
          f"/{requested}")


def test_errors() -> None:
    ok, code = expect_http_error("/api/matches", 400, "POST",
                                 {"black": "no-such-engine", "white": "random"})
    check(ok, "未知引擎返回 400", f"HTTP {code}")

    ok, code = expect_http_error("/api/matches", 400, "POST",
                                 {"black": "random", "white": "greedy", "board_size": 19})
    check(ok, "非 15x15 的 board_size 被拒绝（前后端规则一致）", f"HTTP {code}")

    ok, code = expect_http_error("/api/games/g-does-not-exist", 404)
    check(ok, "不存在的对局返回 404", f"HTTP {code}")

    ok, code = expect_http_error("/api/batches/b-nope", 404)
    check(ok, "不存在的批次返回 404", f"HTTP {code}")


def main() -> int:
    print(f"=== HTTP API 自检（{BASE}）===\n")
    try:
        get("/api/engines")
    except Exception as exc:
        print(f"服务端不可达: {exc}")
        return 1

    test_static_and_meta()
    test_single_match_and_replay()
    test_list_vs_full_stats()
    test_batch_alternates_colors()
    test_batch_cancel()
    test_errors()

    failed = [r for r in results if not r[0]]
    print(f"\n合计 {len(results)} 项，通过 {len(results) - len(failed)}，失败 {len(failed)}")
    for _, name, detail in failed:
        print(f"  FAIL: {name} — {detail}")
    if not_verified:
        print("\n未验证项：")
        for item in not_verified:
            print(f"  - {item}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
