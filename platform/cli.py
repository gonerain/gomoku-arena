"""命令行跑单局（不开浏览器时用）。

    python -m platform.cli --black random --white greedy
    python -m platform.cli --black greedy --white trained --moves
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from platform.match import Match, MatchConfig
    from platform.rules import BOARD_SIZE, PROTOCOL, replay
    from platform.server import ENGINES, Store, launch_spec
else:
    from .match import Match, MatchConfig
    from .rules import BOARD_SIZE, PROTOCOL, replay
    from .server import ENGINES, Store, launch_spec

REASON_TEXT = {
    "five": "五连获胜",
    "draw": "棋盘已满，平局",
    "timeout": "超时判负",
    "crash": "进程崩溃判负",
    "illegal_move": "非法落子判负",
    "bad_output": "输出非协议帧，判负",
    "protocol": "协议违规判负",
    "resign": "主动认输",
    "start_failed": "引擎启动失败",
    "aborted": "对局中止",
}


def build_spec(key: str) -> dict:
    if key in ENGINES:
        return launch_spec(ENGINES[key])
    # 允许直接给可执行文件路径（可跟参数，空格分隔），方便临时接入外部引擎
    parts = key.split()
    path = Path(parts[0])
    if path.exists():
        return launch_spec(
            {"name": path.stem, "version": "v0", "cmd": [sys.executable, str(path), *parts[1:]]}
        )
    raise SystemExit(f"未知引擎: {key!r}，可选 {sorted(ENGINES)} 或一个 .py 文件路径")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="platform.cli", description="五子棋对抗：命令行跑单局")
    parser.add_argument("--black", default="random", help="黑方引擎（名称或 .py 路径）")
    parser.add_argument("--white", default="greedy", help="白方引擎（名称或 .py 路径）")
    parser.add_argument("--move-timeout", type=int, default=1000, help="每步时限 ms")
    parser.add_argument("--game-timeout", type=int, default=120_000, help="整局上限 ms")
    parser.add_argument("--board-size", type=int, default=BOARD_SIZE)
    parser.add_argument("--seed", type=int, default=None, help="随机种子（默认按时间取）")
    parser.add_argument("--moves", action="store_true", help="打印逐手记录")
    parser.add_argument("--board", action="store_true", help="打印终局棋盘")
    parser.add_argument("--save", action="store_true", help="写入 records/games.jsonl")
    parser.add_argument("--protocol", action="store_true", help="打印引擎协议")
    args = parser.parse_args(argv)

    if args.protocol:
        print(PROTOCOL)
        return 0

    config = MatchConfig(
        seed=args.seed,
        board_size=args.board_size,
        move_timeout_ms=args.move_timeout,
        game_timeout_ms=args.game_timeout,
    )
    match = Match(build_spec(args.black), build_spec(args.white), config)
    record = match.play()

    if args.moves or args.board:
        print(f"{'手':>4}  {'方':<5} {'坐标':<6} {'耗时':>7}")
        for m in record["moves"]:
            print(f"{m['n']:>4}  {m['color']:<5} "
                  f"{chr(ord('A') + m['x'])}{m['y'] + 1:<5} {m['ms']:>5}ms")

    if args.board:
        board = replay([[m["color"], m["x"], m["y"]] for m in record["moves"]], record["board_size"])
        print()
        print(board.as_text())
        print("  X=黑 O=白 .=空（自上而下为第 15..1 行）")

    res = record["result"]
    winner = res.get("winner")
    label = f"{winner} ({record[winner]['name']} {record[winner]['version']})" if winner else "无胜者"
    print()
    print(f"对局 {record['game_id']}  黑 {record['black']['name']} vs 白 {record['white']['name']}")
    print(f"结果: {label} · {REASON_TEXT.get(res.get('reason'), res.get('reason'))}")
    if record.get("error"):
        err = record["error"]
        print(f"异常: [{err.get('color')}] {err.get('reason')}: {err.get('message')}")
    print(f"共 {len(record['moves'])} 手，耗时 {record['duration_ms']}ms")

    if args.save:
        store = Store(Path("records/games.jsonl"))
        store.append(record)
        print(f"已写入 {store.path}（共 {len(store.records)} 局）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
