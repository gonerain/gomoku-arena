"""一步贪心引擎：先赢、再堵、否则挑最有价值的位置。

它存在的意义是给 RL 训练提供一个有意义的基线对手 —— 一个随机模型打赢它
说明训练确实在起作用，打不赢说明还没学到东西。

它不是 RL 模型（meta.baseline=true），平台与界面都会把它标注为基线。

    python engines/py_greedy/engine.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # engines/

from engine_loop import BLACK, DIRECTIONS, WHITE, Context, GameView, run  # noqa: E402

# 单方向棋型价值，按连子数与开放端数取分
THREAT_SCORE = {
    (5, 2): 1_000_000, (5, 1): 1_000_000, (5, 0): 1_000_000,   # 成五
    (4, 2): 100_000, (4, 1): 10_000, (4, 0): 2_000,            # 活四 / 冲四
    (3, 2): 5_000, (3, 1): 500, (3, 0): 100,                   # 活三 / 眠三
    (2, 2): 200, (2, 1): 50, (2, 0): 10,
    (1, 2): 20, (1, 1): 5, (1, 0): 1,
}
MAX_CANDIDATES = 24


def line_shape(grid: list[list[int]], size: int, x: int, y: int, dx: int, dy: int, code: int) -> tuple[int, int]:
    """(x,y) 处已有 code 时，该方向上的（连子数, 开放端数）。"""
    run = 1
    open_ends = 0
    for sign in (1, -1):
        cx, cy = x + dx * sign, y + dy * sign
        while 0 <= cx < size and 0 <= cy < size and grid[cy][cx] == code:
            run += 1
            cx += dx * sign
            cy += dy * sign
        if 0 <= cx < size and 0 <= cy < size and grid[cy][cx] == 0:
            open_ends += 1
    return run, open_ends


def score_point(grid: list[list[int]], size: int, x: int, y: int, code: int) -> int:
    total = 0
    for dx, dy in DIRECTIONS:
        run, open_ends = line_shape(grid, size, x, y, dx, dy, code)
        total += THREAT_SCORE.get((min(run, 5), open_ends), 0)
    return total


def move_score(view: GameView, x: int, y: int, ctx: Context) -> int:
    """自己下这里的收益 + 对手下这里的收益（防守权重 0.9）。"""
    grid = view.grid
    size = view.board_size
    grid[y][x] = ctx.code
    mine = score_point(grid, size, x, y, ctx.code)
    grid[y][x] = ctx.opponent_code
    theirs = score_point(grid, size, x, y, ctx.opponent_code)
    grid[y][x] = 0
    return mine + int(theirs * 0.9)


def choose(view: GameView, ctx: Context) -> tuple[int, int]:
    if view.count == 0:
        center = ctx.board_size // 2
        return (center, center)

    candidates = view.neighbors(2) or view.free_cells()

    # 1) 自己能成五，直接赢
    for x, y in candidates:
        if view.would_win(x, y, ctx.code):
            return (x, y)
    # 2) 对手能成五，必须堵
    for x, y in candidates:
        if view.would_win(x, y, ctx.opponent_code):
            return (x, y)

    # 3) 只看离上一手最近的若干候选点，够用且快
    last = view.last_move
    if last and len(candidates) > MAX_CANDIDATES:
        lx, ly = last
        candidates.sort(key=lambda p: abs(p[0] - lx) + abs(p[1] - ly))
        candidates = candidates[:MAX_CANDIDATES]

    best, best_score = candidates[0], -1
    for x, y in candidates:
        value = move_score(view, x, y, ctx)
        if value > best_score:
            best, best_score = (x, y), value
    return best


if __name__ == "__main__":
    run(choose, name="greedy", version="v0",
        meta={"algo": "heuristic-1ply", "desc": "一步贪心：先赢、再堵、否则取邻居"})
