"""端到端测试 B：连续 100 局稳定性与资源泄漏监测。

    python tests/test_stability.py

在一个独立端口的平台实例上跑 100 局，并在过程中周期采样：
  * python 进程总数（含引擎子进程）；
  * 平台进程的句柄数、内存（RSS）、线程数；
  * 崩溃引擎派生的孙进程是否残留。

判定标准：首尾各 20 局的采样均值相比没有持续增长的趋势（不是单点比较，避免抖动误判）。
中途故意让一个引擎崩溃，验证崩溃判负与整棵进程树的清理。

非 Windows 环境下句柄数不可测，会明确标注"未验证"。
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("GOMOKU_TEST_PORT", "8124"))
BASE = f"http://127.0.0.1:{PORT}"
RECORDS = ROOT / "records" / "_e2e_stability.jsonl"
GAMES = int(os.environ.get("GOMOKU_E2E_GAMES", "100"))
MODEL_A = ROOT / "tests" / "models" / "model_a" / "meta.json"
MODEL_B = ROOT / "tests" / "models" / "model_b" / "meta.json"

results: list[tuple[bool, str, str]] = []
not_verified: list[str] = []

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def check(ok: bool, name: str, detail: str = "") -> None:
    ok = bool(ok)
    results.append((ok, name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def skip(name: str, why: str) -> None:
    not_verified.append(f"{name}（{why}）")
    print(f"[SKIP] {name} — {why}")


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
    log = (ROOT / "records" / "_e2e_stability_server.log").open("w", encoding="utf-8")
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


def children_of(pid: int) -> list:
    if psutil is None:
        return []
    try:
        return psutil.Process(pid).children(recursive=True)
    except Exception:
        return []


def sleep_grandchildren() -> list[int]:
    if psutil is None:
        return []
    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(proc.info.get("cmdline") or [])
        if "time.sleep(600)" in cmd:
            found.append(proc.info["pid"])
    return found


def sample(platform_pid: int) -> dict:
    out = {
        "python_procs": 0,
        "all_procs": 0,
        "rss_mb": None,
        "handles": None,
        "threads": None,
    }
    if psutil is None:
        return out
    try:
        out["all_procs"] = len(psutil.pids())
    except Exception:
        pass
    out["python_procs"] = sum(
        1 for p in psutil.process_iter(["name"])
        if (p.info.get("name") or "").lower().startswith("python")
    )
    try:
        me = psutil.Process(platform_pid)
        out["rss_mb"] = round(me.memory_info().rss / 1024 / 1024, 1)
        out["threads"] = me.num_threads()
        if os.name == "nt":
            out["handles"] = me.num_handles()
    except Exception:
        pass
    return out


def main() -> int:
    print(f"=== 端到端测试 B：{GAMES} 局连续稳定性（平台端口 {PORT}）===\n")
    if psutil is None:
        print("未安装 psutil：进程/内存/句柄监测将标注为未验证\n")

    server = start_server()
    samples: list[dict] = []
    baseline_handle = None
    try:
        platform_pid = server.pid
        # 预热 5 局，排除首次分配造成的固定增量
        _, warm = post("/api/batches", {
            "participants": {"black": {"engine": "ckpt_test", "pid": "dsh",
                                       "checkpoint": str(MODEL_A)},
                             "white": {"engine": "ckpt_test", "pid": "codex",
                                       "checkpoint": str(MODEL_B)}},
            "games": 5, "move_timeout_ms": 2000, "game_timeout_ms": 60000, "seed": 100,
        })
        _wait_batch(warm["batch_id"])

        status, batch = post("/api/batches", {
            "participants": {"black": {"engine": "ckpt_test", "pid": "dsh",
                                       "checkpoint": str(MODEL_A)},
                             "white": {"engine": "ckpt_test", "pid": "codex",
                                       "checkpoint": str(MODEL_B)}},
            "games": GAMES, "move_timeout_ms": 2000, "game_timeout_ms": 60000, "seed": 200,
        })
        batch_id = batch["batch_id"]
        check(status == 202, f"{GAMES} 局批量对战已启动", batch_id)

        started = time.time()
        summary = batch
        while time.time() - started < 900:
            summary = get(f"/api/batches/{batch_id}")
            samples.append(sample(platform_pid))
            if summary.get("status") != "running":
                break
            time.sleep(2.0)

        elapsed = time.time() - started
        check(summary.get("status") == "finished", "批量对战正常结束", summary.get("status"))
        check(summary.get("played") == GAMES, f"完成 {GAMES} 局",
              f"{summary.get('played')} 局，用时 {elapsed:.1f}s")
        check(all(g["reason"] == "five" for g in summary["games"]),
              "全部对局都以正常胜负结束（无异常判罚）",
              str(sorted({g["reason"] for g in summary["games"]})))

        # 资源趋势：比较前 25% 与后 25% 采样的均值
        if psutil is None:
            skip("资源泄漏趋势", "未安装 psutil")
        else:
            usable = [s for s in samples if s["rss_mb"] is not None]
            if len(usable) >= 8:
                cut = max(2, len(usable) // 4)
                head, tail = usable[:cut], usable[-cut:]
                for field, label, tolerance in (
                    ("python_procs", "python 进程数", 2),
                    ("handles", "平台句柄数", 40),
                    ("threads", "平台线程数", 12),
                    ("rss_mb", "平台内存 MB", 25),
                ):
                    first = [s[field] for s in head if s[field] is not None]
                    last = [s[field] for s in tail if s[field] is not None]
                    if not first or not last:
                        skip(f"{label}稳定性", "该指标在本环境不可测")
                        continue
                    h_mean, t_mean = statistics.mean(first), statistics.mean(last)
                    check(t_mean - h_mean <= tolerance,
                          f"{label}没有随对局数持续增长",
                          f"前段均值 {h_mean:.1f} → 后段均值 {t_mean:.1f}（容差 {tolerance}）")
            else:
                skip("资源泄漏趋势", f"有效采样不足（{len(usable)}）")

        # 峰谷对比：整批过程中的最大/最小 python 进程数
        if psutil is not None and samples:
            peaks = [s["python_procs"] for s in samples]
            check(max(peaks) <= 8,
                  "过程中并发 python 进程数受控（未出现进程堆积）",
                  f"min={min(peaks)} max={max(peaks)}")

        # 崩溃与超时：故意让一方崩溃，验证判负与整棵进程树清理
        print()
        print("--- 故意制造崩溃 ---")
        stale = sleep_grandchildren()
        check(not stale, "崩溃测试前没有残留的测试孙进程", f"残留={stale}")

        status, snap = post("/api/matches", {
            "participants": {
                "black": {"engine": "ckpt_test", "pid": "healthy",
                          "checkpoint": str(MODEL_B)},
                "white": {"engine": "test_crash", "pid": "victim"},
            },
            "move_timeout_ms": 3000, "game_timeout_ms": 60000, "seed": 900,
        })
        state = _wait_game(snap["game_id"])
        reason = state["result"]["reason"]
        check(state["result"]["winner"] == "black" and reason in ("crash", "timeout"),
              "崩溃引擎被判负且对手获胜", f"winner={state['result']['winner']} reason={reason}")
        time.sleep(1.5)
        leftover_after_crash = sleep_grandchildren()
        check(not leftover_after_crash,
              "崩溃引擎派生的孙进程已被清理",
              f"残留={leftover_after_crash}")

        # 崩溃后平台仍然可用，继续跑一局正常对局
        status, snap = post("/api/matches", {
            "participants": {
                "black": {"engine": "ckpt_test", "pid": "healthy1",
                          "checkpoint": str(MODEL_A)},
                "white": {"engine": "ckpt_test", "pid": "healthy2",
                          "checkpoint": str(MODEL_B)},
            },
            "move_timeout_ms": 2000, "game_timeout_ms": 60000, "seed": 901,
        })
        state = _wait_game(snap["game_id"])
        check(state["result"]["reason"] == "five",
              "崩溃之后平台仍能正常开局", f"reason={state['result']['reason']}")

        # 记录持久化
        lines = [x for x in RECORDS.read_text(encoding="utf-8").splitlines() if x.strip()]
        check(len(lines) >= GAMES, "全部对局已持久化", f"{len(lines)} 行")
        stats = get("/api/stats")
        check(stats["total_games"] >= GAMES, "统计口径覆盖全部对局",
              f"stats.total={stats['total_games']}")
        check(stats.get("records_on_disk", 0) >= GAMES,
              "统计报告了磁盘上的完整条数（内存上限之外的数据仍在文件里）",
              f"on_disk={stats.get('records_on_disk')} in_memory={stats.get('records_in_memory')}")

        if psutil is not None:
            kids = children_of(platform_pid)
            check(len(kids) == 0, "全部结束时平台没有遗留引擎子进程",
                  f"子进程数={len(kids)}")

    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        time.sleep(0.5)

    # 平台退出后也不应该留下引擎进程
    if psutil is not None:
        time.sleep(1.0)
        leftover = sleep_grandchildren()
        check(not leftover, "平台退出后没有残留的测试孙进程", f"残留={leftover}")

    failed = [r for r in results if not r[0]]
    print(f"\n合计 {len(results)} 项，通过 {len(results) - len(failed)}，失败 {len(failed)}")
    for _, name, detail in failed:
        print(f"  FAIL: {name} — {detail}")
    if not_verified:
        print("\n未验证项：")
        for item in not_verified:
            print(f"  - {item}")
    if samples:
        first, last = samples[0], samples[-1]
        print(f"\n采样摘要：{len(samples)} 次")
        print(f"  首次: {first}")
        print(f"  末次: {last}")
    print(f"棋谱文件: {RECORDS}")
    return 1 if failed else 0


def _wait_batch(batch_id: str, timeout: float = 900):
    deadline = time.time() + timeout
    summary = None
    while time.time() < deadline:
        summary = get(f"/api/batches/{batch_id}")
        if summary.get("status") != "running":
            return summary
        time.sleep(0.5)
    return summary


def _wait_game(game_id: str, timeout: float = 120):
    deadline = time.time() + timeout
    state = None
    while time.time() < deadline:
        state = get(f"/api/state?game_id={game_id}")
        if not state.get("running"):
            return state
        time.sleep(0.15)
    return state


if __name__ == "__main__":
    raise SystemExit(main())
