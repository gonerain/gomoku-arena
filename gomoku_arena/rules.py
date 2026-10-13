"""棋盘与规则判定 —— 规则在此冻结，平台其余部分必须与之一致。

冻结规则（free-style 五子棋，非连珠）：
  15x15；黑先；双方轮流落一子于空点；先形成 >=5 连（横、竖、斜）者胜；
  双方均无禁手；无开局规则；棋盘满且无胜者判平局。

不实现连珠(Renju)的长连/双四/双三禁手，也不实现开局型与首手天元限制：
那些属于连珠规则集，与本文冻结的规则不兼容。详见 README 的规则对照表。

坐标约定：x = 列 (0..BOARD_SIZE-1)，y = 行 (0..BOARD_SIZE-1)，原点在左下角。
仅用于显示：x -> A..O，y -> 1..15，天元为 H8。
"""

from __future__ import annotations

BOARD_SIZE = 15

EMPTY = 0
BLACK = 1
WHITE = 2

COLORS = ("black", "white")
COLOR_CODE = {"black": BLACK, "white": WHITE}
CODE_COLOR = {BLACK: "black", WHITE: "white"}

# 四个方向，只取半轴，避免重复扫描
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))

# 星位（仅显示用）
STAR_POINTS = ((3, 3), (11, 3), (3, 11), (11, 11), (7, 7))

# 引擎协议全文（供 `python -m gomoku_arena.cli --protocol` 打印）
PROTOCOL = """
引擎协议（NDJSON：stdin 收 JSON，stdout 发 JSON，每行一个对象；日志一律走 stderr）

  平台 -> 引擎  {"type":"init","game_id":..,"board_size":15,"color":"black","seed":..}
  引擎 -> 平台  {"type":"ready","name":..,"version":..,"meta":{..}}
  平台 -> 引擎  {"type":"turn","moves":[["black",7,7],..],"deadline_ms":1000}
  引擎 -> 平台  {"type":"move","x":8,"y":8}
  平台 -> 引擎  {"type":"gameover","winner":..,"reason":..}
  引擎 -> 平台  {"type":"resign"}            （可选：主动认输）

坐标：x = 列 0..14，y = 行 0..14，原点在左下角。每回合发送完整着法历史，
引擎不需要维护任何对局状态。ready 里的 meta 原样存进棋谱，供记录训练预算。
"""


def other(color: str) -> str:
    return "white" if color == "black" else "black"


def to_coord(x: int, y: int) -> str:
    """(7, 7) -> 'H8'"""
    return f"{chr(ord('A') + x)}{y + 1}"


class Board:
    """棋盘状态。grid[y][x] 取值 0/1/2。"""

    __slots__ = ("size", "grid", "count")

    def __init__(self, size: int = BOARD_SIZE):
        self.size = size
        self.grid = [[EMPTY] * size for _ in range(size)]
        self.count = 0

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def is_empty(self, x: int, y: int) -> bool:
        return self.grid[y][x] == EMPTY

    def is_legal(self, x: int, y: int) -> bool:
        """越界或已占点都返回 False。先判边界，负下标不会绕过检查。"""
        return self.in_bounds(x, y) and self.grid[y][x] == EMPTY

    def place(self, x: int, y: int, code: int) -> None:
        if not self.is_legal(x, y):
            raise ValueError(f"非法落子: ({x},{y})，棋盘尺寸 {self.size}")
        self.grid[y][x] = code
        self.count += 1

    def win_line(self, x: int, y: int, code: int) -> list[list[int]] | None:
        """若 (x,y) 处的 code 形成了 >=5 连，返回构成这条线的所有点（已排序），否则 None。

        注意："five in a row" 判定为 >=5：长连同样获胜，这是五子棋而非连珠。
        """
        if not self.in_bounds(x, y):
            return None
        for dx, dy in DIRECTIONS:
            line = [[x, y]]
            for sign in (1, -1):
                cx, cy = x + dx * sign, y + dy * sign
                while self.in_bounds(cx, cy) and self.grid[cy][cx] == code:
                    line.append([cx, cy])
                    cx += dx * sign
                    cy += dy * sign
            if len(line) >= 5:
                line.sort()
                return line
        return None

    def is_full(self) -> bool:
        return self.count >= self.size * self.size

    def as_text(self) -> str:
        """给日志用的可读棋盘，y 从上到下（第 15 行在最上）。"""
        rows = []
        for y in range(self.size - 1, -1, -1):
            rows.append("".join(".XO"[self.grid[y][x]] for x in range(self.size)))
        return "\n".join(rows)


def replay(moves: list[list], size: int = BOARD_SIZE) -> Board:
    """从着法序列重建棋盘。每个 move 形如 [color, x, y]。"""
    board = Board(size)
    for color, x, y in moves:
        board.place(x, y, COLOR_CODE[color])
    return board


def last_win_line(moves: list[list], size: int = BOARD_SIZE) -> list[list[int]] | None:
    """从完整着法序列里找出获胜的五连，供回放/统计使用（独立于裁判内部状态）。"""
    board = Board(size)
    for color, x, y in moves:
        code = COLOR_CODE[color]
        board.place(x, y, code)
        line = board.win_line(x, y, code)
        if line:
            return line
    return None
