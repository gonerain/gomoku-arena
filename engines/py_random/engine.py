"""最小可用引擎：随机落子。它就是接口的最小实现，也是平台的自检引擎。

    python engines/py_random/engine.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # engines/

from engine_loop import Context, GameView, run  # noqa: E402


def choose(view: GameView, ctx: Context) -> tuple[int, int]:
    if view.count == 0:
        center = ctx.board_size // 2
        return (center, center)
    return ctx.rng.choice(view.free_cells())


if __name__ == "__main__":
    run(choose, name="random", version="v0", meta={"algo": "uniform", "desc": "随机落子"})
