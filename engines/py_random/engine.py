"""最小可用引擎：随机落子。它就是接口的最小实现，也是平台的自检引擎。

它不是 RL 模型，而是**基线**（meta.baseline=true）：一个随机策略的成绩是训练效果的
下界参照，平台与界面都会把它标注为基线，避免被误读成"模型"。

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
