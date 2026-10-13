"""跑完整回归：7 个套件，逐个执行并把输出汇总到一行。

    python tests/run_all.py

单元/接口套件（规则、身份、健壮性、API）需要能自行结束；
E2E 套件（双模型、稳定性、观战）会各自拉起独立端口的平台实例。
API 套件需要 8090 上已有平台实例（本脚本不会替你启动它）。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITES = [
    ("规则", "test_rules.py", False),
    ("身份与推理接口", "test_identity.py", False),
    ("健壮性/句柄/进程树", "test_robustness.py", False),
    ("HTTP API", "test_api.py", True),
    ("E2E-A 双模型 20 局", "test_dual_models.py", False),
    ("E2E-B 100 局稳定性", "test_stability.py", False),
    ("E2E-C 批量实时观战", "test_batch_spectate.py", False),
]


def server_reachable() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen("http://127.0.0.1:8090/api/engines", timeout=5):
            return True
    except Exception:
        return False


def main() -> int:
    import os

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    rows = []
    for label, script, needs_server in SUITES:
        if needs_server and not server_reachable():
            rows.append((label, "SKIP", "8090 上没有运行中的平台实例"))
            print(f"[SKIP] {label} — 需要先启动 python -m gomoku_arena.server --port 8090")
            continue
        started = time.time()
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tests" / script)],
            cwd=str(ROOT), env=env, capture_output=True, text=True, encoding="utf-8",
            errors="replace",
        )
        elapsed = time.time() - started
        summary = ""
        for line in reversed((proc.stdout or "").splitlines()):
            if "合计" in line:
                summary = line.strip()
                break
        failures = [
            line.strip() for line in (proc.stdout or "").splitlines() if line.startswith("[FAIL]")
        ]
        status = "PASS" if proc.returncode == 0 else "FAIL"
        rows.append((label, status, summary or f"exit={proc.returncode}"))
        print(f"[{status}] {label} ({elapsed:.1f}s) — {summary}")
        for line in failures[:6]:
            print(f"        {line}")
        if proc.returncode != 0 and not summary:
            print((proc.stderr or "")[-800:])

    print()
    failed = [r for r in rows if r[1] == "FAIL"]
    skipped = [r for r in rows if r[1] == "SKIP"]
    print(f"套件合计 {len(rows)}：通过 {len(rows) - len(failed) - len(skipped)}，"
          f"失败 {len(failed)}，跳过 {len(skipped)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
