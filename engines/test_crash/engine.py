"""测试用引擎：派生子进程后崩溃 —— 用于验证崩溃判负与整棵进程树被清理。

它会启动一个长时间 sleep 的孙子进程，然后自己退出。
如果平台的清理不彻底，这个孙子进程会残留。
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine_loop import Context, GameView, run  # noqa: E402

CHILD = sys.executable


def choose(view: GameView, ctx: Context) -> tuple[int, int]:
    # 派生一个 600 秒的孙子进程，写出它的 pid 供测试检查
    child = subprocess.Popen([CHILD, "-c", "import time; time.sleep(600)"])
    print(f"[crasher] 已派生孙进程 pid={child.pid}", file=sys.stderr)
    raise RuntimeError("故意崩溃")


if __name__ == "__main__":
    run(choose, name="crasher", version="v0", meta={"algo": "crash"})
