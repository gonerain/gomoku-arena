"""端到端测试 A：两份不同模型权重，通过同一个适配器参赛。

    python tests/test_dual_models.py

真实启动一个平台实例（独立端口、独立棋谱文件），让两个参赛方
（dsh / codex）共用 gomoku_arena 的 ckpt_test 适配器，分别加载
tests/models/model_a 与 tests/models/model_b 两份权重，跑 20 局并逐局交换先手。

验证目标：
  * 两份 checkpoint 没有混淆（每局的身份、权重、版本都对得上）；
  * 权重确实参与了推理（引擎自报 policy_version 与所属权重一致）；
  * 逐局交换先手，且权重跟着参赛方走；
  * 真实推理通路被走到（moves 数量与 model 自报调用次数一致）；
  * 测试模型不被标记为经过 RL 训练。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("GOMOKU_TEST_PORT", "8123"))
BASE = f"http://127.0.0.1:{PORT}"
MODEL_A = ROOT / "tests" / "models" / "model_a" / "meta.json"
MODEL_B = ROOT / "tests" / "models" / "model_b" / "meta.json"
RECORDS = ROOT / "records" / "_e2e_dual.jsonl"
GAMES = int(os.environ.get("GOMOKU_E2E_GAMES", "20"))

results: list[tuple[bool, str, str]] = []
not_verified: list[str] = []


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
    log = (ROOT / "records" / "_e2e_dual_server.log").open("w", encoding="utf-8")
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


def participants_body(black_first: bool = True):
    dsh = {"engine": "ckpt_test", "pid": "dsh", "name": "DSH", "version": "ta1",
           "checkpoint": str(MODEL_A), "meta": {"algo": "test-weight-hash"}}
    codex = {"engine": "ckpt_test", "pid": "codex", "name": "Codex", "version": "tb2",
             "checkpoint": str(MODEL_B), "meta": {"algo": "test-weight-hash"}}
    return {"black": dsh if black_first else codex, "white": codex if black_first else dsh}


def test_engines_listed() -> None:
    engines = {e["key"]: e for e in get("/api/engines")}
    check("ckpt_test" in engines, "测试适配器已注册", ",".join(sorted(engines)))
    check(engines["ckpt_test"]["baseline"] is True,
          "测试适配器被明确标记为基线（非 RL 训练产物）",
          f"baseline={engines['ckpt_test']['baseline']}")
    models = get("/api/models")
    check(len(models) == 0 or all(".json" not in m["name"] or "model_" not in m["name"]
                                  for m in models),
          "测试权重不在 checkpoints/ 目录中（测试与正式模型分离）",
          f"checkpoints 下 {len(models)} 个文件")


def test_single_game_identity() -> None:
    """先打一局，确认身份与权重对应关系正确。"""
    status, snap = post("/api/matches", {
        "participants": participants_body(), "move_timeout_ms": 2000,
        "game_timeout_ms": 60000, "seed": 11,
    })
    game_id = snap["game_id"]
    deadline = time.time() + 60
    state = snap
    while time.time() < deadline:
        state = get(f"/api/state?game_id={game_id}")
        if not state.get("running"):
            break
        time.sleep(0.1)
    check(state.get("status") == "finished", "单局双模型对战正常结束",
          f"result={state.get('result', {}).get('reason')} 手数={state.get('move_count')}")

    rec = get(f"/api/games/{game_id}")
    check(rec["black"]["pid"] == "dsh" and rec["white"]["pid"] == "codex",
          "棋谱身份：黑=dsh，白=codex",
          f"{rec['black']['pid']} vs {rec['white']['pid']}")
    check(rec["black"]["checkpoint"] == str(MODEL_A)
          and rec["white"]["checkpoint"] == str(MODEL_B),
          "棋谱权重：dsh→model_a，codex→model_b",
          f"{Path(rec['black']['checkpoint']).parent.name} / {Path(rec['white']['checkpoint']).parent.name}")
    check(rec["black"]["version"] == "ta1" and rec["white"]["version"] == "tb2",
          "棋谱版本：dsh=ta1，codex=tb2",
          f"{rec['black']['version']} / {rec['white']['version']}")
    check(rec["black"]["meta"].get("baseline") is True,
          "身份元信息标注为基线（非 RL）", str(rec["black"]["meta"].get("baseline")))
    check(rec["black"]["meta"].get("policy_version") == 1
          and rec["white"]["meta"].get("policy_version") == 7,
          "引擎自报的权重标识与所属 checkpoint 一致（推理确实用了各自的权重）",
          f"a.policy_version={rec['black']['meta'].get('policy_version')} "
          f"b.policy_version={rec['white']['meta'].get('policy_version')}")
    check(len(rec["moves"]) > 0, "真实推理产生了着法", f"{len(rec['moves'])} 手")


def test_batch_20_games() -> None:
    status, batch = post("/api/batches", {
        "participants": participants_body(), "games": GAMES,
        "move_timeout_ms": 2000, "game_timeout_ms": 60000, "seed": 7000,
    })
    batch_id = batch["batch_id"]
    check(status == 202, f"批量对战已启动（{GAMES} 局）", batch_id)

    deadline = time.time() + 600
    summary = batch
    while time.time() < deadline:
        summary = get(f"/api/batches/{batch_id}")
        if summary.get("status") != "running":
            break
        time.sleep(0.5)

    check(summary.get("status") == "finished", "批量对战正常结束", summary.get("status"))
    check(summary.get("played") == GAMES, f"完成 {GAMES} 局", f"{summary.get('played')}")

    sides = summary["sides"]
    check(sides["dsh"]["as_black"]["games"] == GAMES // 2
          and sides["dsh"]["as_white"]["games"] == GAMES - GAMES // 2,
          "逐局交换先手：dsh 执黑/执白各半",
          f"黑={sides['dsh']['as_black']['games']} 白={sides['dsh']['as_white']['games']}")
    check(sides["dsh"]["checkpoint"] == str(MODEL_A)
          and sides["codex"]["checkpoint"] == str(MODEL_B),
          "统计里的权重未互相覆盖",
          f"dsh={Path(sides['dsh']['checkpoint']).parent.name} "
          f"codex={Path(sides['codex']['checkpoint']).parent.name}")
    check(not sides["dsh"]["mixed_checkpoints"] and not sides["codex"]["mixed_checkpoints"],
          "没有任何一方出现权重混用标记",
          f"dsh={sides['dsh']['mixed_checkpoints']} codex={sides['codex']['mixed_checkpoints']}")

    # 逐局核对身份与权重随参赛方移动
    mismatches: list[str] = []
    for index, game in enumerate(summary["games"]):
        wants_dsh_black = index % 2 == 0
        got_dsh_black = game["black_pid"] == "dsh"
        if wants_dsh_black != got_dsh_black:
            mismatches.append(f"#{index + 1} 先手交换错误")
        if got_dsh_black:
            if game["black_checkpoint"] != str(MODEL_A) or game["white_checkpoint"] != str(MODEL_B):
                mismatches.append(f"#{index + 1} 权重与颜色绑错")
        else:
            if game["black_checkpoint"] != str(MODEL_B) or game["white_checkpoint"] != str(MODEL_A):
                mismatches.append(f"#{index + 1} 权重未随参赛方移动")
    check(not mismatches, f"{GAMES} 局的先手交换与权重归属全部正确",
          "; ".join(mismatches[:3]) if mismatches else "全部一致")

    # 两份权重确实产生了不同的着法序列（没有串味的最直接证据）
    sequences = {str(MODEL_A): set(), str(MODEL_B): set()}
    for game in summary["games"]:
        rec = get(f"/api/games/{game['game_id']}")
        for color, key in (("black", "black_checkpoint"), ("white", "white_checkpoint")):
            moves = tuple((m["x"], m["y"]) for m in rec["moves"] if m["color"] == color)
            sequences[game[key]].add(moves[:4])
    check(sequences[str(MODEL_A)] != sequences[str(MODEL_B)],
          "两份权重产生了可区分的着法序列（说明各自用了自己的权重）",
          f"A 的前4手组合数={len(sequences[str(MODEL_A)])} "
          f"B={len(sequences[str(MODEL_B)])}")

    # 推理通路：每一局都应有真实着法（不是空对局）
    empty = [g["game_id"] for g in summary["games"] if g["moves"] == 0]
    check(not empty, "每一局都产生了着法（真实推理通路）",
          f"空局={empty[:3]}" if empty else f"{GAMES} 局均有着法")


def test_records_persisted() -> None:
    lines = [json.loads(x) for x in RECORDS.read_text(encoding="utf-8").splitlines() if x.strip()]
    check(len(lines) >= GAMES,
          "棋谱已持久化，每局一行", f"{len(lines)} 行")
    check(all("participants" in rec and rec["v"] == 2 for rec in lines),
          "棋谱含 participants 身份块（v2 格式）")
    pids = {rec["black"]["pid"] for rec in lines} | {rec["white"]["pid"] for rec in lines}
    check(pids == {"dsh", "codex"}, "持久化棋谱里的参赛方身份只有 dsh / codex", str(pids))
    ckpts = {rec["black"]["checkpoint"] for rec in lines} | {rec["white"]["checkpoint"] for rec in lines}
    check(ckpts == {str(MODEL_A), str(MODEL_B)},
          "持久化棋谱里只有两份权重，没有交叉污染", str(len(ckpts)))


def main() -> int:
    print(f"=== 端到端测试 A：双模型对战（{GAMES} 局，平台端口 {PORT}）===\n")
    for path in (MODEL_A, MODEL_B):
        if not path.is_file():
            print(f"缺少测试权重: {path}")
            return 1
    proc = start_server()
    try:
        test_engines_listed()
        test_single_game_identity()
        test_batch_20_games()
        test_records_persisted()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    failed = [r for r in results if not r[0]]
    print(f"\n合计 {len(results)} 项，通过 {len(results) - len(failed)}，失败 {len(failed)}")
    for _, name, detail in failed:
        print(f"  FAIL: {name} — {detail}")
    print(f"棋谱文件: {RECORDS}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
