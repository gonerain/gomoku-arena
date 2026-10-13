"""规则单元测试：横/竖/斜五连、边界、长连、未获胜、非法落子、平局。

规则冻结为 free-style 五子棋（15x15，黑先，>=5 连获胜，双方无禁手），
本文件是这套规则的唯一验收依据；平台其余部分必须与它一致。

    python tests/test_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gomoku_arena.rules import (  # noqa: E402
    BLACK,
    BOARD_SIZE,
    STAR_POINTS,
    WHITE,
    Board,
    last_win_line,
    other,
    replay,
    to_coord,
)

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    results.append((ok, name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def make(cells, code=BLACK):
    """按坐标列表放同色棋子，返回最后一个落点。"""
    board = Board()
    last = None
    for x, y in cells:
        board.place(x, y, code)
        last = (x, y)
    return board, last


def test_overview() -> None:
    check(BOARD_SIZE == 15, "棋盘尺寸冻结为 15x15", f"BOARD_SIZE={BOARD_SIZE}")
    check(STAR_POINTS == ((3, 3), (11, 3), (3, 11), (11, 11), (7, 7)),
          "星位定义固定（含天元 H8）")
    check(other("black") == "white" and other("white") == "black", "对手颜色互换")
    check(to_coord(7, 7) == "H8" and to_coord(0, 0) == "A1" and to_coord(14, 14) == "O15",
          "坐标显示 A1..O15 与天元 H8")


def test_horizontal() -> None:
    board, last = make([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
    line = board.win_line(*last, BLACK)
    check(line is not None and len(line) == 5, "横向五连获胜",
          f"line={line}")


def test_vertical() -> None:
    board, last = make([(5, 3), (5, 4), (5, 5), (5, 6), (5, 7)])
    line = board.win_line(*last, BLACK)
    check(line is not None and len(line) == 5, "纵向五连获胜", f"line={line}")


def test_diagonal_up() -> None:
    board, last = make([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
    line = board.win_line(*last, BLACK)
    check(line is not None and len(line) == 5, "斜向（↗）五连获胜", f"line={line}")


def test_diagonal_down() -> None:
    board, last = make([(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)])
    line = board.win_line(*last, BLACK)
    check(line is not None and len(line) == 5, "斜向（↘）五连获胜", f"line={line}")


def test_win_detected_from_middle_stone() -> None:
    """中间落子也应能识别整条线，而不是只从末尾数。"""
    board, _ = make([(3, 8), (4, 8), (5, 8), (6, 8)])
    board.place(7, 8, WHITE)
    board.place(2, 8, BLACK)  # 补成 2..6 共 5 连
    line = board.win_line(2, 8, BLACK)
    check(line is not None and len(line) == 5, "从中间补成的五连被识别", f"line={line}")


def test_boundary_wins() -> None:
    cases = {
        "贴左边界纵向": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        "贴右边界横向": [(10, 14), (11, 14), (12, 14), (13, 14), (14, 14)],
        "贴右下角斜向": [(10, 10), (11, 11), (12, 12), (13, 13), (14, 14)],
        "贴左上角反向斜向": [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)],
    }
    for name, cells in cases.items():
        board, last = make(cells)
        line = board.win_line(*last, BLACK)
        check(line is not None and len(line) == 5, f"边界情况：{name}", f"line={line}")


def test_overline_wins() -> None:
    """free-style 规则：>=5 即胜，长连（6 连及以上）同样获胜。"""
    board, last = make([(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5)])
    line = board.win_line(*last, BLACK)
    check(line is not None and len(line) == 6, "长连（6 连）判胜且线包含全部 6 子",
          f"line={line}")


def test_overline_wins_for_both_colors() -> None:
    for code, label in ((BLACK, "黑"), (WHITE, "白")):
        board, last = make([(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5)], code=code)
        line = board.win_line(*last, code)
        check(line is not None and len(line) == 7, f"长连判胜对{label}方同样成立",
              f"len={len(line) if line else 0}")


def test_no_win_cases() -> None:
    board, last = make([(0, 0), (1, 0), (2, 0), (3, 0)])
    check(board.win_line(*last, BLACK) is None, "四连不算获胜")

    board, last = make([(0, 0), (1, 0), (2, 0), (4, 0), (5, 0)])
    check(board.win_line(*last, BLACK) is None,
          "被断开的五个点不算获胜（中间有空格）", "0,1,2,_,4,5")
    # 形状：X X O X X X  —— 白子把黑子切成 2 连和 3 连，任一侧都不足 5
    board, last = make([(0, 0), (1, 0), (3, 0), (4, 0), (5, 0)])
    board.place(2, 0, WHITE)
    check(board.win_line(3, 0, BLACK) is None and board.win_line(1, 0, BLACK) is None,
          "被对方棋子截断不算获胜", "X X O X X X")

    board = Board()
    check(board.win_line(7, 7, BLACK) is None, "空棋盘没有获胜线")
    board.place(7, 7, BLACK)
    check(board.win_line(7, 7, BLACK) is None, "单子不获胜")


def test_illegal_moves() -> None:
    board = Board()
    board.place(7, 7, BLACK)

    check(not board.is_legal(7, 7), "已占点不是合法落点")
    check(not board.is_legal(-1, 0), "x 越界（负）不是合法落点")
    check(not board.is_legal(0, -1), "y 越界（负）不是合法落点")
    check(not board.is_legal(15, 0), "x 越界（>=size）不是合法落点")
    check(not board.is_legal(0, 15), "y 越界（>=size）不是合法落点")
    check(board.is_legal(0, 0) and board.is_legal(14, 14), "棋盘四角内为空点时可落子")

    for x, y, name in ((7, 7, "已占点"), (-1, 0, "负坐标"), (15, 15, "越界坐标")):
        try:
            board.place(x, y, WHITE)
            check(False, f"非法落子应抛异常：{name}", "竟然成功了")
        except ValueError as exc:
            check(True, f"非法落子抛 ValueError：{name}", str(exc))

    check(board.count == 1, "非法落子不改变棋子计数", f"count={board.count}")


def test_draw_full_board() -> None:
    """棋盘被填满且无五连 → 平局。构造一个刻意不含五连的填充方式。"""
    board = Board()
    # 按 (x + 2*y) % 4 < 2 染色，保证任何方向上连续同色都不超过 2
    code_for = lambda x, y: BLACK if ((x + 2 * y) % 4) < 2 else WHITE
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            board.place(x, y, code_for(x, y))

    check(board.is_full() and board.count == BOARD_SIZE * BOARD_SIZE,
          "棋盘填满后 is_full 为真", f"count={board.count}")

    winners = []
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            code = board.grid[y][x]
            if board.win_line(x, y, code):
                winners.append((x, y))
    check(not winners, "该填充方式下确实不存在五连（平局成立）", f"意外获胜点={winners[:3]}")


def test_replay_and_helpers() -> None:
    moves = [["black", 7, 7], ["white", 7, 8], ["black", 8, 7], ["white", 8, 8],
             ["black", 9, 7], ["white", 9, 8], ["black", 10, 7], ["white", 10, 8],
             ["black", 11, 7]]
    board = replay(moves)
    check(board.count == 9 and board.grid[7][7] == BLACK and board.grid[8][7] == WHITE,
          "replay 能从着法序列重建棋盘", f"count={board.count}")
    line = last_win_line(moves)
    check(line is not None and len(line) == 5, "last_win_line 能从着法序列还原获胜五连",
          f"line={line}")
    check(last_win_line([["black", 7, 7], ["white", 8, 8]]) is None,
          "无胜局的 last_win_line 返回 None")

    text = replay(moves).as_text()
    check(len(text.splitlines()) == BOARD_SIZE, "as_text 输出 15 行",
          f"行数={len(text.splitlines())}")


def main() -> int:
    print("=== 规则单元测试（free-style 五子棋）===\n")
    test_overview()
    test_horizontal()
    test_vertical()
    test_diagonal_up()
    test_diagonal_down()
    test_win_detected_from_middle_stone()
    test_boundary_wins()
    test_overline_wins()
    test_overline_wins_for_both_colors()
    test_no_win_cases()
    test_illegal_moves()
    test_draw_full_board()
    test_replay_and_helpers()

    failed = [r for r in results if not r[0]]
    print(f"\n合计 {len(results)} 项，通过 {len(results) - len(failed)}，失败 {len(failed)}")
    for _, name, detail in failed:
        print(f"  FAIL: {name} — {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
