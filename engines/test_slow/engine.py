"""测试用引擎：每步严重超时 —— 用于验证超时判负与进程树清理。"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine_loop import Context, GameView, run  # noqa: E402


def choose(view: GameView, ctx: Context) -> tuple[int, int]:
    time.sleep(30)  # 远超单步时限
    return (0, 0)


if __name__ == "__main__":
    run(choose, name="slowpoke", version="v0", meta={"algo": "sleep"})
