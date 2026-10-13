"""健壮性自检：单步超时 / 全局超时归责 / 崩溃 / 非法落子 / 脏输出 / 认输 /
启动失败 / 进程树清理 / checkpoint 诚实性。

    python tests/test_robustness.py

进程树清理这条最关键：测试引擎会派生一个 600 秒的孙进程再崩溃，
平台必须在判负的同时把孙进程也收拾掉。

环境说明：进程树清理依赖 Windows Job Object。若在非 Windows 上运行，
本文件会明确报告"未验证"，而不是假装通过。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engines"))

from gomoku_arena.engine_host import (  # noqa: E402
    EngineProc,
    JobHandle,
    _create_kill_on_close_job,
)
from gomoku_arena.match import Match, MatchConfig  # noqa: E402
from gomoku_arena.server import launch_spec  # noqa: E402

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

results: list[tuple[bool, str, str]] = []
not_verified: list[str] = []

TRAINED_ENGINE = ROOT / "engines" / "py_trained" / "engine.py"


def check(ok: bool, name: str, detail: str = "") -> None:
    ok = bool(ok)
    results.append((ok, name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def skip(name: str, why: str) -> None:
    not_verified.append(f"{name}（{why}）")
    print(f"[SKIP] {name} — {why}")


def spec_for(script: str, *args: str) -> dict:
    path = ROOT / "engines" / script / "engine.py"
    return launch_spec(
        {
            "name": script,
            "version": "v0",
            "cmd": [sys.executable, str(path), *args],
            "meta": {"baseline": True},
        }
    )


def inline_spec(name: str, code: str) -> dict:
    return launch_spec(
        {"name": name, "version": "v0", "cmd": [sys.executable, "-c", code],
         "meta": {"baseline": True}}
    )


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


def find_sleep_grandchildren() -> list[int]:
    """找出测试引擎派生的 `time.sleep(600)` 孙进程。"""
    pids: list[int] = []
    try:
        import psutil

        for proc in psutil.process_iter(["pid", "cmdline"]):
            cmd = " ".join(proc.info.get("cmdline") or [])
            if "time.sleep(600)" in cmd:
                pids.append(proc.info["pid"])
    except Exception:
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine"],
            capture_output=True, text=True, check=False,
        ).stdout
        for line in out.splitlines():
            if "time.sleep(600)" in line:
                match = re.search(r"(\d+)\s*$", line.strip())
                if match:
                    pids.append(int(match.group(1)))
    return pids


# ---------- 正常对局与规则判罚 ----------

def test_normal_match() -> None:
    rec = play(spec_for("py_greedy"), spec_for("py_random"), move_timeout_ms=3000)
    seen: set[tuple[int, int]] = set()
    legal = True
    for move in rec["moves"]:
        if (move["x"], move["y"]) in seen or not (0 <= move["x"] < 15 and 0 <= move["y"] < 15):
            legal = False
        seen.add((move["x"], move["y"]))
    alternates = all(
        move["color"] == ("black" if i % 2 == 0 else "white")
        for i, move in enumerate(rec["moves"])
    )
    check(legal and alternates and len(rec["moves"]) > 0,
          "正常对局：着法合法且黑白交替", f"{len(rec['moves'])} 手")
    check(rec["result"]["reason"] in ("five", "draw"),
          "正常对局：产生明确结果", rec["result"]["reason"])
    check("game_timeout_ms" in rec and "move_timeout_ms" in rec,
          "棋谱记录了两类时限配置", f"move={rec['move_timeout_ms']} game={rec['game_timeout_ms']}")


def test_per_move_timeout() -> None:
    rec = play(spec_for("test_slow"), spec_for("py_greedy"))
    check(rec["result"]["winner"] == "white" and rec["result"]["reason"] == "timeout"
          and rec["error"]["color"] == "black",
          "单步超时：判该方负并正确归因",
          f"winner={rec['result']['winner']} reason={rec['result']['reason']}")


def test_global_timeout_is_neutral() -> None:
    """全局墙钟上限是平台护栏，绝不能被归责于当前落子方。"""
    started = time.monotonic()
    rec = play(spec_for("test_slow"), spec_for("py_greedy"),
               move_timeout_ms=5000, game_timeout_ms=2500)
    elapsed = time.monotonic() - started
    check(rec["result"]["winner"] is None and rec["result"]["reason"] == "game_timeout",
          "全局超时：判为无胜负中止，不归责任何一方",
          f"winner={rec['result']['winner']} reason={rec['result']['reason']}")
    check(rec.get("error", {}).get("color") is None
          and "不归责" in (rec.get("error", {}).get("message") or ""),
          "全局超时：错误信息注明不归责", (rec.get("error") or {}).get("message", ""))
    check(elapsed < 4.5,
          "全局超时：墙钟上限能真正终止当前这一步",
          f"实际耗时 {elapsed:.2f}s（上限 2.5s，单步时限 5s）")


def test_illegal_occupied() -> None:
    rec = play(spec_for("test_illegal", "occupied"), spec_for("py_greedy"))
    check(rec["result"]["winner"] == "white" and rec["result"]["reason"] == "illegal_move",
          "非法落子判负：重复占点", (rec.get("error") or {}).get("message", ""))


def test_illegal_out_of_bounds() -> None:
    rec = play(spec_for("test_illegal", "out"), spec_for("py_greedy"))
    check(rec["result"]["winner"] == "white" and rec["result"]["reason"] == "illegal_move",
          "非法落子判负：坐标越界", (rec.get("error") or {}).get("message", ""))


def test_bad_output() -> None:
    code = ("import time;print('this is not json');import sys;sys.stdout.flush();time.sleep(30)")
    rec = play(inline_spec("noisy", code), spec_for("py_greedy"))
    check(rec["result"]["winner"] == "white" and rec["result"]["reason"] == "bad_output",
          "脏输出判负：stdout 输出非协议内容", (rec.get("error") or {}).get("message", ""))


def test_protocol_violation() -> None:
    """回合里回一个错误类型的帧 → 协议违规判负。"""
    code = (
        "import sys,json,time;"
        "sys.stdin.readline();"
        "print(json.dumps({'type':'ready','name':'wrong','version':'v0'}),flush=True);"
        "sys.stdin.readline();"
        "print(json.dumps({'type':'hello'}),flush=True);"
        "time.sleep(30)"
    )
    rec = play(inline_spec("wrongframe", code), spec_for("py_greedy"))
    check(rec["result"]["winner"] == "white" and rec["result"]["reason"] == "protocol",
          "协议违规判负：回合收到非 move 帧", (rec.get("error") or {}).get("message", ""))


def test_resign() -> None:
    code = (
        "import sys,json,time;"
        "sys.stdin.readline();"
        "print(json.dumps({'type':'ready','name':'resigner','version':'v0'}),flush=True);"
        "sys.stdin.readline();"
        "print(json.dumps({'type':'resign'}),flush=True);"
        "time.sleep(30)"
    )
    rec = play(inline_spec("resigner", code), spec_for("py_greedy"))
    check(rec["result"]["winner"] == "white" and rec["result"]["reason"] == "resign",
          "主动认输：判该方负", rec["result"]["reason"])


def test_start_failed() -> None:
    """真不存在的情况才触发 START_FAILED。

    注意 Windows 上 `python <不存在的脚本>` 不会让 Popen 抛错（CreateProcess 只管
    启动 exe），所以这里用的是"可执行文件不存在"与"工作目录不存在"这两种
    CreateProcess 必然失败的形态。
    """
    missing_exe = ROOT / "engines" / "definitely-missing-python.exe"
    spec = launch_spec(
        {"name": "missing", "version": "v0", "cmd": [str(missing_exe), "engine.py"],
         "meta": {"baseline": True}}
    )
    rec = play(spec, spec_for("py_greedy"))
    check(rec["result"]["winner"] == "white" and rec["result"]["reason"] == "start_failed"
          and rec["error"]["color"] == "black",
          "引擎启动失败（可执行文件不存在）：判该方负并归因",
          f"reason={rec['result']['reason']} · {(rec.get('error') or {}).get('message', '')[:60]}")

    spec = launch_spec(
        {
            "name": "baddir", "version": "v0",
            "cmd": [sys.executable, str(ROOT / "engines" / "py_random" / "engine.py")],
            "cwd": str(ROOT / "no_such_working_dir"),
            "meta": {"baseline": True},
        }
    )
    rec = play(spec, spec_for("py_greedy"))
    check(rec["result"]["winner"] == "white" and rec["result"]["reason"] == "start_failed",
          "引擎启动失败（工作目录不存在）：判该方负并归因",
          f"reason={rec['result']['reason']}")


def test_engine_script_missing_is_reported() -> None:
    """脚本文件不存在时 python 会立刻退出：平台必须明确判负，不能当成正常对局。"""
    spec = launch_spec(
        {
            "name": "nofile", "version": "v0",
            "cmd": [sys.executable, str(ROOT / "engines" / "no_such_engine_dir" / "engine.py")],
            "meta": {"baseline": True},
        }
    )
    rec = play(spec, spec_for("py_greedy"))
    check(rec["result"]["winner"] == "white"
          and rec["result"]["reason"] in ("start_failed", "crash")
          and len(rec["moves"]) == 0,
          "引擎脚本不存在：明确判负且未产生任何着法",
          f"reason={rec['result']['reason']}")


def test_crash_verdict() -> None:
    rec = play(spec_for("test_crash"), spec_for("py_greedy"), move_timeout_ms=3000)
    check(rec["result"]["winner"] == "white" and rec["result"]["reason"] in ("crash", "timeout"),
          "崩溃/挂起：对局正常终止并归因", f"reason={rec['result']['reason']}")


# ---------- checkpoint 诚实性 ----------

def test_checkpoint_missing_fails_loudly() -> None:
    spec = launch_spec(
        {
            "name": "trained", "version": "v0",
            "cmd": [sys.executable, str(TRAINED_ENGINE)],
            "env": {"GOMOKU_MODEL": "checkpoints/definitely-not-here.pt"},
            "meta": {"baseline": False},
            "checkpoint": "checkpoints/definitely-not-here.pt",
        }
    )
    rec = play(spec, spec_for("py_greedy"))
    check(rec["result"]["winner"] == "white"
          and rec["result"]["reason"] in ("start_failed", "crash"),
          "指定不存在的 checkpoint：显式失败，不静默退回贪心",
          f"reason={rec['result']['reason']}")
    check(len(rec["moves"]) == 0,
          "指定不存在的 checkpoint：没有产生任何着法（未伪装成会下棋的模型）",
          f"手数={len(rec['moves'])}")
    message = (rec.get("error") or {}).get("message", "")
    check("退出码 2" in message,
          "指定不存在的 checkpoint：判负原因写明退出码 2（便于一眼看出是模型加载失败）",
          message)


def test_checkpoint_unimplemented_fails_loudly() -> None:
    """checkpoint 文件存在但接入模板未实现 _load() → 仍必须显式失败。"""
    fake = ROOT / "checkpoints" / "_test_fake_model.pt"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_bytes(b"not a real model")
    try:
        spec = launch_spec(
            {
                "name": "trained", "version": "v0",
                "cmd": [sys.executable, str(TRAINED_ENGINE)],
                "env": {"GOMOKU_MODEL": str(fake)},
                "meta": {"baseline": False},
                "checkpoint": str(fake),
            }
        )
        rec = play(spec, spec_for("py_greedy"))
        check(rec["result"]["winner"] == "white" and len(rec["moves"]) == 0,
              "checkpoint 存在但 _load() 未实现：显式失败，不冒充 RL 模型",
              f"reason={rec['result']['reason']} 手数={len(rec['moves'])}")
    finally:
        fake.unlink(missing_ok=True)


def test_trained_demo_mode_is_labelled() -> None:
    """未指定 checkpoint 的演示模式必须能对局，且明确标注 baseline=true。"""
    spec = launch_spec(
        {
            "name": "trained", "version": "v0",
            "cmd": [sys.executable, str(TRAINED_ENGINE)],
            "env": {"GOMOKU_MODEL": ""},
            "meta": {"baseline": False},
        }
    )
    rec = play(spec, spec_for("py_random"), move_timeout_ms=3000)
    black = rec["black"]
    check(rec["result"]["reason"] in ("five", "draw") and len(rec["moves"]) > 0
          and black["meta"].get("baseline") is True
          and black["meta"].get("algo") == "heuristic-demo",
          "演示模式：可正常对局，但 meta 明确标注为基线（非 RL）",
          f"algo={black['meta'].get('algo')} baseline={black['meta'].get('baseline')} "
          f"手数={len(rec['moves'])}")


def test_trained_exit_code_contract() -> None:
    """直接跑引擎进程，确认退出码契约：无效 checkpoint → 退出码 2 + 明确原因。"""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["GOMOKU_MODEL"] = "checkpoints/definitely-not-here.pt"
    proc = subprocess.run(
        [sys.executable, str(TRAINED_ENGINE)],
        input='{"type":"init","game_id":"t","board_size":15,"color":"black","seed":1}\n',
        capture_output=True, text=True, env=env, timeout=30,
    )
    check(proc.returncode == 2 and "无法作为 RL 模型运行" in proc.stderr,
          "退出码契约：无效 checkpoint 以退出码 2 结束并说明原因",
          f"returncode={proc.returncode}")
    check(proc.stdout.strip() == "",
          "退出码契约：失败时不在 stdout 输出任何协议帧", repr(proc.stdout[:60]))


# ---------- 进程与清理 ----------

def test_job_object_available() -> None:
    job = _create_kill_on_close_job()
    check(job is not None, "Windows Job Object 可用（kill-on-close）",
          "" if job else "不可用，将退回 best-effort 清理")
    if job is None:
        return
    check(isinstance(job, JobHandle) and not job.closed,
          "Job 返回带所有权标记的句柄对象", f"type={type(job).__name__}")
    first = job.close()
    second = job.close()
    check(first is True and second is False and job.closed,
          "同一句柄只被真正关闭一次（第二次调用直接返回，不再触碰内核对象）",
          f"first={first} second={second} closed={job.closed}")


def test_handle_reuse_is_safe() -> None:
    """句柄值会被系统复用：不能靠"句柄数值"记忆是否关过。

    这里交错创建/关闭多个 Job，若实现仍按数值记账，后创建的 Job 只要拿到刚被
    回收的句柄值就会被误判为"已关闭"从而漏关（表现为句柄泄漏）。
    """
    if os.name != "nt":
        skip("句柄值复用安全", "非 Windows 环境")
        return
    handles = []
    for _ in range(6):
        job = _create_kill_on_close_job()
        if job is None:
            check(False, "句柄值复用安全", "无法创建 Job")
            return
        handles.append(job)
        job.close()
    values = [h.value for h in handles]
    duplicates = len(values) != len(set(values))
    check(duplicates,
          "确实观察到了句柄值复用（说明本测试有意义）",
          f"values={values}")
    check(all(h.closed for h in handles), "所有 Job 都被各自正确关闭")


def test_handle_count_stable() -> None:
    """连续多局后本进程句柄数不应持续增长。"""
    if os.name != "nt" or psutil is None:
        skip("句柄数稳定性", "需要 Windows + psutil")
        return
    me = psutil.Process()

    def spawn_cycle(rounds: int) -> None:
        for _ in range(rounds):
            proc = EngineProc(spec_for("py_random"), "handles")
            proc.start()
            proc.send({"type": "init", "game_id": "t", "board_size": 15,
                       "color": "black", "seed": 1})
            proc.recv(5000)
            proc.close()

    spawn_cycle(3)  # 预热：排除首次分配带来的固定增量
    before = me.num_handles()
    spawn_cycle(8)
    after = me.num_handles()
    check(after <= before + 4,
          "8 轮引擎启停后句柄数没有持续增长",
          f"{before} -> {after}（允许 +4 抖动）")


def test_close_is_idempotent() -> None:
    proc = EngineProc(spec_for("py_random"), "idem")
    proc.start()
    proc.send({"type": "init", "game_id": "t", "board_size": 15, "color": "black", "seed": 1})
    proc.recv(5000)
    proc.close()
    proc.close()  # 必须安全
    check(True, "EngineProc.close 可重复调用且不抛异常")


def test_crash_and_tree_cleanup() -> None:
    if os.name != "nt":
        skip("进程树清理（孙进程）", "非 Windows 环境，Job Object 路径无法验证")
        return

    for stale in find_sleep_grandchildren():
        try:
            import psutil

            psutil.Process(stale).kill()
        except Exception:
            pass

    proc = EngineProc(spec_for("test_crash"), "probe")
    proc.start()
    proc.send({"type": "init", "game_id": "t", "board_size": 15, "color": "black", "seed": 1})
    ready = proc.recv(5000)
    check(ready.get("type") == "ready", "崩溃引擎能正常握手")

    proc.send({"type": "turn", "moves": [], "deadline_ms": 800})
    time.sleep(1.2)  # 等它派生孙进程并崩溃

    grandchildren = find_sleep_grandchildren()
    if not grandchildren:
        check(False, "进程树清理：捕获到孙进程", "未能定位测试孙进程，本项无法判定")
        proc.close()
        return
    pid = grandchildren[0]
    check(pid_alive(pid), "进程树清理：孙进程已被派生", f"pid={pid}")

    proc.close()
    time.sleep(1.0)
    check(not pid_alive(pid), "进程树清理：孙进程随对局结束被清除", f"pid={pid}")


def main() -> int:
    print("=== 五子棋平台健壮性自检 ===\n")
    for fn in (
        test_job_object_available,
        test_handle_reuse_is_safe,
        test_handle_count_stable,
        test_normal_match,
        test_per_move_timeout,
        test_global_timeout_is_neutral,
        test_illegal_occupied,
        test_illegal_out_of_bounds,
        test_bad_output,
        test_protocol_violation,
        test_resign,
        test_start_failed,
        test_engine_script_missing_is_reported,
        test_checkpoint_missing_fails_loudly,
        test_checkpoint_unimplemented_fails_loudly,
        test_trained_demo_mode_is_labelled,
        test_trained_exit_code_contract,
        test_close_is_idempotent,
        test_crash_and_tree_cleanup,
        test_crash_verdict,
    ):
        fn()

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
