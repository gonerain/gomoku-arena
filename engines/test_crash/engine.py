"""测试用引擎：派生子进程后崩溃 —— 用于验证崩溃判负与整棵进程树被清理。

它会启动一个长时间 sleep 的孙子进程，然后自己退出。
如果平台的清理不彻底，这个孙子进程会残留。

默认第一手就崩；设 GOMOKU_TEST_CRASH_AFTER=N 可以等到第 N 手再崩，
便于在长批量对战中途制造崩溃。
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine_loop import Context, GameView, run  # noqa: E402

CHILD = sys.executable
CRASH_AFTER = int(os.environ.get("GOMOKU_TEST_CRASH_AFTER", "0") or 0)
_PLIES = 0


def choose(view: GameView, ctx: Context) -> tuple[int, int]:
    global _PLIES
    _PLIES += 1
    if _PLIES <= CRASH_AFTER:
        # 先正常走几手
        for x in range(view.board_size):
            for y in range(view.board_size):
                if view.grid[y][x] == 0:
                    return (x, y)
    # 派生一个 600 秒的孙子进程，写出它的 pid 供测试检查
    child = subprocess.Popen([CHILD, "-c", "import time; time.sleep(600)"])
    print(f"[crasher] 已派生孙进程 pid={child.pid}", file=sys.stderr)
    raise RuntimeError("故意崩溃")


if __name__ == "__main__":
    run(choose, name="crasher", version="v0", meta={"algo": "crash", "baseline": True})
