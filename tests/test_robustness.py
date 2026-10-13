"""健壮性自检：超时 / 崩溃 / 非法落子 / 进程树清理。

    python tests/test_robustness.py

进程树清理这条最关键：测试引擎会派生一个 600 秒的孙进程再崩溃，
平台必须在判负的同时把孙进程也收拾掉。
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engines"))

from platform.engine_host import EngineProc, _create_kill_on_close_job  # noqa: E402
from platform.match import Match, MatchConfig  # noqa: E402
from platform.server import launch_spec  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))


def spec_for(script: str, *args: str) -> dict:
    path = ROOT / "engines" / script / "engine.py"
    return launch_spec({"name": script, "version": "v0", "cmd": [sys.executable, str(path), *args]})


def play(black: dict, white: dict, **cfg) -> dict:
    config = MatchConfig(**{"move_timeout_ms": 800, "game_timeout_ms": 60_000, **cfg})
    return Match(black, white, config).play()


def pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, check=False
        )
        return str(pid) in out.stdout


def test_timeout() -> None:
    rec = play(spec_for("test_slow"), spec_for("py_greedy"))
    ok = (rec["result"]["winner"] == "white" and rec["result"]["reason"] == "timeout"
          and rec["error"]["color"] == "black")
    record("超时判负：归因到超时的一方", ok,
           f"winner={rec['result']['winner']} reason={rec['result']['reason']}")


def test_illegal_occupied() -> None:
    rec = play(spec_for("test_illegal", "occupied"), spec_for("py_greedy"))
    ok = (rec["result"]["winner"] == "white" and rec["result"]["reason"] == "illegal_move")
    record("非法落子判负：重复占点", ok, rec["error"]["message"] if rec.get("error") else "")


def test_illegal_out_of_bounds() -> None:
    rec = play(spec_for("test_illegal", "out"), spec_for("py_greedy"))
    ok = (rec["result"]["winner"] == "white" and rec["result"]["reason"] == "illegal_move")
    record("非法落子判负：坐标越界", ok, rec["error"]["message"] if rec.get("error") else "")


def test_crash_and_tree_cleanup() -> None:
    """崩溃的引擎派生了孙进程，判负之后孙子进程也必须消失。"""
    proc = EngineProc(spec_for("test_crash"), "probe")
    proc.start()
    import json as _json

    proc.send({"type": "init", "game_id": "t", "board_size": 15, "color": "black", "seed": 1})
    ready = proc.recv(5000)
    record("崩溃引擎能正常握手", ready.get("type") == "ready")

    # 让它跑一步：它会派生孙进程然后自己崩掉
    proc.send({"type": "turn", "moves": [], "deadline_ms": 800})
    time.sleep(1.2)  # 等它派生 + 崩溃

    grandchild_pid = None
    grandchild_pids = []
    # 从 stderr 拿不到（已被转发），改为直接扫描：孙进程的命令行特征是 sleep(600)
    if psutil_available():
        import psutil

        for p in psutil.process_iter(["pid", "cmdline"]):
            cmd = " ".join(p.info.get("cmdline") or [])
            if "time.sleep(600)" in cmd:
                grandchild_pids.append(p.info["pid"])
    else:  # 退回 wmic
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine"],
            capture_output=True, text=True, check=False,
        ).stdout
        for line in out.splitlines():
            if "time.sleep(600)" in line:
                m = re.search(r"(\d+)\s*$", line.strip())
                if m:
                    grandchild_pids.append(int(m.group(1)))

    if not grandchild_pids:
        record("进程树清理：捕获到孙进程", False, "未能定位到测试孙进程，本项无法判定")
    else:
        grandchild_pid = grandchild_pids[0]
        record("进程树清理：孙进程已被派生", pid_alive(grandchild_pid), f"pid={grandchild_pid}")

    proc.close()
    time.sleep(1.0)

    if grandchild_pid is not None:
        record("进程树清理：孙进程已随对局结束被清除", not pid_alive(grandchild_pid),
               f"pid={grandchild_pid}")


def test_crash_verdict() -> None:
    """崩溃应被判负（本轮可能先超时，取决于孙进程是否持有管道）。"""
    rec = play(spec_for("test_crash"), spec_for("py_greedy"), move_timeout_ms=3000)
    reason = rec["result"]["reason"]
    ok = rec["result"]["winner"] == "white" and reason in ("crash", "timeout")
    record("崩溃/挂起判负：对局正常结束并归因", ok, f"reason={reason}")


def test_job_object_available() -> None:
    job = _create_kill_on_close_job()
    ok = job is not None
    from platform.engine_host import _close_handle

    _close_handle(job)
    record("Windows Job Object 可用（进程树强清理手段）", ok,
           "" if ok else "不可用，将退回 best-effort 清理")


def psutil_available() -> bool:
    try:
        import psutil  # noqa: F401

        return True
    except Exception:
        return False


def test_state_machine() -> None:
    """正常对局：着法序列合法、结果自洽。"""
    rec = play(spec_for("py_greedy"), spec_for("py_random"), move_timeout_ms=3000)
    seen = set()
    legal = True
    for m in rec["moves"]:
        if (m["x"], m["y"]) in seen or not (0 <= m["x"] < 15 and 0 <= m["y"] < 15):
            legal = False
        seen.add((m["x"], m["y"]))
    colors_alternate = all(
        m["color"] == ("black" if i % 2 == 0 else "white") for i, m in enumerate(rec["moves"])
    )
    record("对局状态机：着法合法且黑白交替", legal and colors_alternate,
           f"{len(rec['moves'])} 手")
    record("对局状态机：产生明确胜者", rec["result"]["winner"] in ("black", "white", None)
           and rec["result"]["reason"] in ("five", "draw"), rec["result"]["reason"])


def main() -> int:
    print("=== 五子棋平台健壮性自检 ===\n")
    test_job_object_available()
    test_state_machine()
    test_timeout()
    test_illegal_occupied()
    test_illegal_out_of_bounds()
    test_crash_and_tree_cleanup()
    test_crash_verdict()

    failed = [r for r in results if r[0] == FAIL]
    print(f"\n合计 {len(results)} 项，通过 {len(results) - len(failed)}，失败 {len(failed)}")
    for _, name, detail in failed:
        print(f"  FAIL: {name} — {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
