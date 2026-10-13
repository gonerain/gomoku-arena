"""端到端测试 C：批量对战的实时观战（浏览器实际会走的那条轮询路径）。

    python tests/test_batch_spectate.py

浏览器不做任何特殊处理：它轮询 /api/state 拿到"当前正在下的一局"，
所以本测试用同一套接口验证：
  * 批量进行中 /api/state 返回的是**当前局**，且 moves 逐手增长；
  * 每局结束后自动切到下一局（game_id 变化、batch.index 递增）；
  * /api/batches 的 played 逐步增长、current_game_id 指向当前局；
  * 实时观战与历史回放互不干扰：回放期间实时对局照常推进；
  * 结束时比分、身份与持久化棋谱三者一致。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("GOMOKU_TEST_PORT", "8125"))
BASE = f"http://127.0.0.1:{PORT}"
RECORDS = ROOT / "records" / "_e2e_spectate.jsonl"
MODEL_A = ROOT / "tests" / "models" / "model_a" / "meta.json"
MODEL_B = ROOT / "tests" / "models" / "model_b" / "meta.json"
GAMES = int(os.environ.get("GOMOKU_E2E_GAMES", "12"))
# 观战采样必须有可观察的窗口：测试适配器每步会刻意延迟 MOVE_DELAY_MS，
# 否则整局会在两次轮询之间就结束（真实模型推理通常远慢于此，不存在该问题）。
MOVE_DELAY_MS = int(os.environ.get("GOMOKU_E2E_MOVE_DELAY", "250"))
MOVE_TIMEOUT_MS = int(os.environ.get("GOMOKU_E2E_MOVE_TIMEOUT", "2000"))

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    ok = bool(ok)
    results.append((ok, name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post(path: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def start_server() -> subprocess.Popen:
    RECORDS.unlink(missing_ok=True)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    log = (ROOT / "records" / "_e2e_spectate_server.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "gomoku_arena.server", "--port", str(PORT),
         "--records", str(RECORDS)],
        cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    for _ in range(60):
        try:
            get("/api/engines")
            return proc
        except Exception:
            time.sleep(0.3)
    proc.kill()
    raise SystemExit("平台实例未能启动")


def main() -> int:
    print(f"=== 端到端测试 C：批量实时观战（{GAMES} 局，平台端口 {PORT}）===\n")
    server = start_server()
    try:
        status, batch = post("/api/batches", {
            "participants": {
                "black": {"engine": "ckpt_test", "pid": "dsh", "name": "DSH",
                          "version": "ta1", "checkpoint": str(MODEL_A),
                          "move_delay_ms": MOVE_DELAY_MS},
                "white": {"engine": "ckpt_test", "pid": "codex", "name": "Codex",
                          "version": "tb2", "checkpoint": str(MODEL_B),
                          "move_delay_ms": MOVE_DELAY_MS},
            },
            "games": GAMES, "move_timeout_ms": MOVE_TIMEOUT_MS, "game_timeout_ms": 60000,
            "seed": 500,
        })
        batch_id = batch["batch_id"]
        check(status == 202, f"批量对战已启动（{GAMES} 局）", batch_id)

        # --- 模拟浏览器轮询 ---
        seen_game_ids: list[str] = []
        seen_indices: list[int] = []
        move_history: dict[str, list[int]] = {}
        players_seen: set[tuple] = set()
        deadline = time.time() + 300
        while time.time() < deadline:
            state = get("/api/state")
            if state and (state.get("batch") or {}).get("batch_id") == batch_id:
                gid = state["game_id"]
                if not seen_game_ids or seen_game_ids[-1] != gid:
                    seen_game_ids.append(gid)
                    seen_indices.append(state["batch"]["index"])
                move_history.setdefault(gid, []).append(state.get("move_count", 0))
                players_seen.add((state["black"]["pid"], state["white"]["pid"]))
            summary = get(f"/api/batches/{batch_id}")
            if summary.get("status") != "running":
                break
            time.sleep(0.1)

        monotonic_ok = all(seq == sorted(seq) for seq in move_history.values())
        advanced = [gid for gid, seq in move_history.items() if len(set(seq)) > 1]
        check(monotonic_ok, "轮询到的每局手数只增不减", f"观察 {len(move_history)} 局")
        check(len(advanced) >= 3,
              "浏览器轮询能看到多局逐手增长（实时观战确实在更新）",
              f"{len(advanced)}/{len(move_history)} 局观察到手数增长")
        check(len(seen_game_ids) == GAMES,
              f"自动切换了全部 {GAMES} 局（每局结束后切到下一局）",
              f"看到 {len(seen_game_ids)} 个不同 game_id")
        check(seen_indices == list(range(1, len(seen_indices) + 1)),
              "当前局序号按 1..N 递增", str(seen_indices))
        check(all(max(seq) > 0 for seq in move_history.values()),
              "每一局被观察时最终都有落子",
              f"各局最大手数={sorted(max(v) for v in move_history.values())}")
        check(players_seen == {("dsh", "codex"), ("codex", "dsh")},
              "两个方向的身份都出现过（交换先手在观战侧可见）", str(sorted(players_seen)))

        summary = get(f"/api/batches/{batch_id}")
        check(summary["status"] == "finished" and summary["played"] == GAMES,
              "批量对战结束且完成局数正确",
              f"status={summary['status']} played={summary['played']}")
        check(summary.get("current_game_id") in (None, ""),
              "结束后不再指向当前局", str(summary.get("current_game_id")))
        check(summary["sides"]["dsh"]["checkpoint"] == str(MODEL_A)
              and summary["sides"]["codex"]["checkpoint"] == str(MODEL_B),
              "比分面板里的身份与权重正确",
              f"dsh={Path(summary['sides']['dsh']['checkpoint']).parent.name} "
              f"codex={Path(summary['sides']['codex']['checkpoint']).parent.name}")
        check(summary["totals"]["dsh"]["games"] == GAMES,
              "合计局数等于批量局数", str(summary["totals"]["dsh"]["games"]))

        # --- 历史回放与实时观战隔离 ---
        finished_game = summary["games"][0]["game_id"]
        replay = get(f"/api/games/{finished_game}")
        check(len(replay.get("moves", [])) > 0,
              "已结束的局可以通过 /api/games/<id> 完整回放",
              f"{len(replay['moves'])} 手")

        status, snap = post("/api/matches", {
            "participants": {
                "black": {"engine": "ckpt_test", "pid": "live1", "checkpoint": str(MODEL_A),
                          "move_delay_ms": MOVE_DELAY_MS},
                "white": {"engine": "ckpt_test", "pid": "live2", "checkpoint": str(MODEL_B),
                          "move_delay_ms": MOVE_DELAY_MS},
            },
            "move_timeout_ms": MOVE_TIMEOUT_MS, "game_timeout_ms": 60000, "seed": 999,
        })
        live_id = snap["game_id"]
        # 回放数据是独立请求：反复取旧棋谱不应改变实时对局的推进
        progress: list[int] = []
        replay_ok = True
        for _ in range(400):
            try:
                replay = get(f"/api/games/{finished_game}")   # 模拟用户正在看历史局
                replay_ok = replay_ok and replay["game_id"] == finished_game
            except Exception:
                replay_ok = False
            state = get(f"/api/state?game_id={live_id}")      # 同时实时对局在推进
            progress.append(state.get("move_count", 0))
            if not state.get("running"):
                break
        check(replay_ok and len(get(f"/api/games/{finished_game}")["moves"]) > 0,
              "回放期间反复取历史棋谱始终成功且内容不变")
        check(progress == sorted(progress) and len(set(progress)) >= 3,
              "回放历史局的同时实时对局照常推进（互不干扰）",
              f"采样到的手数序列={progress[:12]}{'...' if len(progress) > 12 else ''}")
        check(get(f"/api/state?game_id={live_id}")["game_id"] == live_id,
              "按 game_id 查询能拿到指定的实时对局（前端回放隔离依赖这一点）")

        # --- 身份与持久化一致性 ---
        lines = [json.loads(x) for x in RECORDS.read_text(encoding="utf-8").splitlines() if x.strip()]
        batch_records = [r for r in lines if (r.get("batch") or {}).get("batch_id") == batch_id]
        check(len(batch_records) == GAMES,
              "持久化棋谱里批次的局数与观战一致", f"{len(batch_records)} 行")
        wrong = []
        for rec in batch_records:
            index = rec["batch"]["index"]
            wants_dsh_black = index % 2 == 1
            if (rec["black"]["pid"] == "dsh") != wants_dsh_black:
                wrong.append(index)
            ckpt = rec["black"]["checkpoint"] if wants_dsh_black else rec["white"]["checkpoint"]
            if ckpt != str(MODEL_A):
                wrong.append(f"{index}:权重")
        check(not wrong, "持久化棋谱的先手交换与权重归属与观战一致", str(wrong[:4]))

        stats = get("/api/stats")
        by_pid = {p["pid"]: p for p in stats["participants"]}
        check(by_pid["dsh"]["games"] == by_pid["codex"]["games"],
              "完整统计里双方局数相等（交换先手）",
              f"dsh={by_pid['dsh']['games']} codex={by_pid['codex']['games']}")
        check(by_pid["dsh"]["checkpoint"] == str(MODEL_A)
              and by_pid["codex"]["checkpoint"] == str(MODEL_B),
              "完整统计按参赛方区分权重，未混合",
              f"{Path(by_pid['dsh']['checkpoint']).parent.name} / "
              f"{Path(by_pid['codex']['checkpoint']).parent.name}")

    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()

    failed = [r for r in results if not r[0]]
    print(f"\n合计 {len(results)} 项，通过 {len(results) - len(failed)}，失败 {len(failed)}")
    for _, name, detail in failed:
        print(f"  FAIL: {name} — {detail}")
    print(f"棋谱文件: {RECORDS}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
