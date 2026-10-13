"""测试用引擎：返回非法坐标 —— 用于验证非法落子判负。

两种模式（用 argv[1] 选择）：
    occupied  落在已占点
    out      越界
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine_loop import Context, GameView, run  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "occupied"


def choose(view: GameView, ctx: Context) -> tuple[int, int]:
    if MODE == "out":
        return (-1, 99)
    if view.history:
        return view.history[0]  # 重复下第一手
    return (0, 0)


if __name__ == "__main__":
    run(choose, name=f"illegal-{MODE}", version="v0", meta={"algo": "illegal", "mode": MODE})
