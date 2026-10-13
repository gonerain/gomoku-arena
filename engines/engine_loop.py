"""引擎侧协议实现（复制这个文件即可开始写自己的引擎）。

用法：在任意目录放一份 engine.py，调用 run(choose) 即可。

    from engine_loop import run          # 或把本文件复制到你的引擎目录旁
    def choose(moves, board_size, ctx):
        ...
        return (x, y)
    run(choose, name="mybot", version="v1", meta={"algo": "ppo", "episodes": 12000})

协议（NDJSON）：stdin 收 JSON，stdout 发 JSON，每行一个对象。
调试日志请走 stderr —— 任何写到 stdout 的非协议内容都会被判为脏输出。

  收 {"type":"init","game_id":..,"board_size":15,"color":"black","seed":..}
  发 {"type":"ready","name":..,"version":..,"meta":{..}}
  收 {"type":"turn","moves":[["black",7,7],..],"deadline_ms":1000}
  发 {"type":"move","x":8,"y":8}
  收 {"type":"gameover","winner":..,"reason":..}   然后退出

坐标：x = 列 0..14，y = 行 0..14，原点在左下角。
每回合都会收到完整着法历史，引擎不需要维护任何跨回合状态。
"""

from __future__ import annotations

import json
import random
import sys
from typing import Callable

BLACK, WHITE = 1, 2
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


class GameView:
    """把协议里的 moves 历史整理成引擎好用的形式。只读。

    可用接口：board_size / grid / history / count / last_move /
    in_bounds() / is_empty() / free_cells() / neighbors(distance) / would_win()。
    RL 侧做推理时通常用 is_empty() 或 legal_mask() 屏蔽非法点。
    """

    def __init__(self, board_size: int, moves: list[list]):
        self.board_size = board_size
        self.moves = moves
        self.grid = [[0] * board_size for _ in range(board_size)]
        self.history: list[tuple[int, int]] = []
        for index, (color, x, y) in enumerate(moves):
            self.grid[y][x] = BLACK if color == "black" else WHITE
            self.history.append((x, y))
        self.count = len(moves)

    @property
    def last_move(self) -> tuple[int, int] | None:
        return self.history[-1] if self.history else None

    def in_bounds(self, x: int, y: int) -> bool:
        """坐标是否在棋盘内。真实推理返回的落点必须先过这一关。"""
        return 0 <= x < self.board_size and 0 <= y < self.board_size

    def is_empty(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.grid[y][x] == 0

    def free_cells(self) -> list[tuple[int, int]]:
        size = self.board_size
        return [(x, y) for y in range(size) for x in range(size) if self.grid[y][x] == 0]

    def neighbors(self, distance: int = 2) -> list[tuple[int, int]]:
        """已有棋子附近 distance 格内的空点，用于裁剪动作空间。"""
        size = self.board_size
        if not self.history:
            center = size // 2
            return [(center, center)]
        out: set[tuple[int, int]] = set()
        for x, y in self.history:
            for dy in range(-distance, distance + 1):
                for dx in range(-distance, distance + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size and self.grid[ny][nx] == 0:
                        out.add((nx, ny))
        return sorted(out)

    def would_win(self, x: int, y: int, code: int) -> bool:
        """假设 code 落在 (x,y)，是否形成 >=5 连。"""
        for dx, dy in DIRECTIONS:
            run = 1
            for sign in (1, -1):
                cx, cy = x + dx * sign, y + dy * sign
                while 0 <= cx < self.board_size and 0 <= cy < self.board_size and self.grid[cy][cx] == code:
                    run += 1
                    cx += dx * sign
                    cy += dy * sign
            if run >= 5:
                return True
        return False


class Context:
    """一次对局的元信息，传给 choose()。"""

    def __init__(self, init_frame: dict):
        self.game_id = init_frame.get("game_id")
        self.board_size = int(init_frame.get("board_size", 15))
        self.color = init_frame.get("color", "black")
        self.seed = int(init_frame.get("seed", 0))
        self.rng = random.Random(self.seed)
        self.code = BLACK if self.color == "black" else WHITE
        self.opponent_code = WHITE if self.code == BLACK else BLACK


def run(choose: Callable[[GameView, Context], tuple[int, int]],
        name: str = "engine", version: str = "v0", meta: dict | None = None) -> None:
    """引擎主循环。choose(view, ctx) 返回 (x, y)。"""
    out = sys.stdout
    line = sys.stdin.readline()
    if not line:
        return
    ctx = Context(json.loads(line))
    out.write(json.dumps({"type": "ready", "name": name, "version": version,
                          "meta": meta or {}}, ensure_ascii=False) + "\n")
    out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        frame = json.loads(line)
        ftype = frame.get("type")
        if ftype == "gameover":
            print(f"[{name}] 对局结束: {frame.get('winner')} / {frame.get('reason')}", file=sys.stderr)
            return
        if ftype != "turn":
            print(f"[{name}] 忽略未知帧: {ftype}", file=sys.stderr)
            continue

        view = GameView(ctx.board_size, [[c, x, y] for c, x, y in frame.get("moves", [])])
        x, y = choose(view, ctx)
        out.write(json.dumps({"type": "move", "x": int(x), "y": int(y)}) + "\n")
        out.flush()
